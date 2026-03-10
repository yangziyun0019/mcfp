// File: orientation_dataset_cli.cpp
// Purpose: Generate anchor-based orientation reachability datasets from a prepared position dataset.
// Usage: ros2 run reachability_cli orientation_dataset_cli --config <yaml>

#include "reachability_cli/model_loader.h"

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>
#include <hdf5.h>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

#include <moveit/collision_detection/collision_common.h>
#include <moveit/kdl_kinematics_plugin/kdl_kinematics_plugin.h>
#include <moveit/robot_model/joint_model_group.h>
#include <moveit/robot_state/robot_state.h>
#include <random_numbers/random_numbers.h>

#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
struct Options
{
  std::string config_path;
};

struct Config
{
  std::string robot_name;
  std::string urdf_path;
  std::string srdf_path;
  std::string group_name;
  std::string base_link;
  std::string ee_link;
  std::vector<std::string> joint_names;

  std::string base_h5_path;

  size_t anchors_total = 2000;
  int candidate_min_count = 3;
  std::array<int, 3> anchor_bins_xyz{ 8, 8, 8 };
  std::array<int, 3> anchor_bins_scr{ 6, 4, 4 };
  size_t anchor_min_quota_xyz = 1;
  size_t anchor_min_quota_scr = 2;
  std::string anchor_radius_mode = "linf";
  double anchor_eps = 0.02;
  double anchor_alpha = 0.8;
  double anchor_lambda = 0.25;
  double anchor_gamma = 2.0;
  double anchor_beta = 0.5;
  double anchor_p = 2.0;
  double anchor_n0 = 100.0;
  int anchor_neighbor_mode = 6;

  int coarse_min = 4096;
  int coarse_max = 8192;
  int knn_k = 16;
  double d_max = 0.40;
  std::string d_max_mode = "adaptive";
  double d_max_factor = 3.0;
  double d_max_clip_min = 0.25;
  double d_max_clip_max = 0.60;
  double delta_boundary = 0.01;
  size_t max_boundary_edges = 2048;
  size_t min_boundary_edges = 256;
  std::vector<double> shell_deltas{ 0.01, 0.02, 0.04, 0.08 };
  int refine_count = 0;
  double refine_sigma = 0.03;
  int seeds_per_anchor = 32;
  double warm_start_mix = 0.7;
  bool scramble_per_anchor = false;
  uint64_t sobol_seed = 0;
  int q_ref_size = 64;
  int q_ref_pool = 4096;
  int method3_boundary_max = 0;
  int method3_boundary_min = 0;
  double method3_min_abs_phi = 0.0;
  double method3_phi_margin = 0.0;

  double ik_timeout = 0.005;
  int ik_csr_trials = 2;
  int ik_random_trials = 1;
  int bisect_csr_trials = 2;
  int bisect_random_trials = 2;
  double search_discretization = 0.005;

  int threads = 0;

  int debug_max_anchors = 1;
  int debug_start_anchor = 0;

  std::string output_path;
  size_t hdf5_chunk = 100000;
  bool write_joint = true;
  bool write_method = true;
  bool write_csr = true;
  bool flush_per_anchor = true;
  bool stream_csr = true;
  bool anchor_only = false;
};

struct Grid
{
  size_t nx = 0;
  size_t ny = 0;
  size_t nz = 0;
  double voxel_size = 0.005;
  Eigen::Vector3d origin{ 0.0, 0.0, 0.0 };

  size_t index(size_t x, size_t y, size_t z) const
  {
    return (x * ny + y) * nz + z;
  }

  Eigen::Vector3d center(size_t x, size_t y, size_t z) const
  {
    return origin + voxel_size *
                       Eigen::Vector3d(static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5,
                                       static_cast<double>(z) + 0.5);
  }

  size_t size() const
  {
    return nx * ny * nz;
  }
};

struct Anchor
{
  size_t idx = 0;
  float s_v = 0.0f;
  float c_v = 0.0f;
  float g_v = 0.0f;
  uint32_t n_seed = 0;
  Eigen::Vector3f pos{ 0.0f, 0.0f, 0.0f };
  double weight = 0.0;
};

struct Seed
{
  Eigen::Quaterniond quat;
  std::vector<double> joint;
};

struct OrientationSample
{
  Eigen::Quaterniond quat;
  bool reachable = false;
  std::vector<double> joint;
};

struct OutputSample
{
  Eigen::Quaterniond quat;
  float phi = 0.0f;
  int8_t label = 0;
  uint8_t method = 0;
  std::vector<float> joint;
};

bool parseArgs(int argc, char** argv, Options& options)
{
  for (int i = 1; i < argc; ++i)
  {
    std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc)
    {
      options.config_path = argv[++i];
    }
    else if (arg == "--help" || arg == "-h")
    {
      return false;
    }
    else
    {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }
  return !options.config_path.empty();
}

void printUsage()
{
  std::cerr << "Usage: orientation_dataset_cli --config <path>\n";
}

template <typename T>
T getScalar(const YAML::Node& node, const std::string& key, const T& fallback)
{
  if (node && node[key])
  {
    return node[key].as<T>();
  }
  return fallback;
}

template <typename T>
T getRequiredScalar(const YAML::Node& node, const std::string& key)
{
  if (!node || !node[key])
  {
    throw std::runtime_error("Missing required config key: " + key);
  }
  return node[key].as<T>();
}

std::vector<std::string> getStringList(const YAML::Node& node, const std::string& key)
{
  std::vector<std::string> out;
  if (node && node[key])
  {
    for (const auto& item : node[key])
    {
      out.push_back(item.as<std::string>());
    }
  }
  return out;
}

std::vector<double> getDoubleList(const YAML::Node& node, const std::string& key,
                                  const std::vector<double>& fallback)
{
  if (node && node[key])
  {
    std::vector<double> out;
    for (const auto& item : node[key])
    {
      out.push_back(item.as<double>());
    }
    return out;
  }
  return fallback;
}

std::vector<int> getIntList(const YAML::Node& node, const std::string& key, const std::vector<int>& fallback)
{
  if (node && node[key])
  {
    std::vector<int> out;
    for (const auto& item : node[key])
    {
      out.push_back(item.as<int>());
    }
    return out;
  }
  return fallback;
}

double percentile(std::vector<double>& values, double q)
{
  if (values.empty())
  {
    return 0.0;
  }
  q = std::min(1.0, std::max(0.0, q));
  const size_t idx = static_cast<size_t>(std::llround(q * static_cast<double>(values.size() - 1)));
  std::nth_element(values.begin(), values.begin() + static_cast<long>(idx), values.end());
  return values[idx];
}

Config loadConfig(const std::string& path)
{
  YAML::Node root = YAML::LoadFile(path);
  YAML::Node robot = root["robot"] ? root["robot"] : root;
  YAML::Node input = root["input"] ? root["input"] : root;
  YAML::Node anchors = root["anchors"] ? root["anchors"] : root;
  YAML::Node sampling = root["sampling"] ? root["sampling"] : root;
  YAML::Node ik = root["ik"] ? root["ik"] : root;
  YAML::Node output = root["output"] ? root["output"] : root;
  YAML::Node debug = root["debug"] ? root["debug"] : root;
  YAML::Node threads = root["threads"] ? root["threads"] : root;

  Config cfg;
  cfg.robot_name = getScalar<std::string>(robot, "name", "robot");
  cfg.urdf_path = getRequiredScalar<std::string>(robot, "urdf");
  cfg.srdf_path = getRequiredScalar<std::string>(robot, "srdf");
  cfg.group_name = getRequiredScalar<std::string>(robot, "group_name");
  cfg.base_link = getRequiredScalar<std::string>(robot, "base_link");
  cfg.ee_link = getRequiredScalar<std::string>(robot, "ee_link");
  cfg.joint_names = getStringList(robot, "joint_names");

  cfg.base_h5_path = getRequiredScalar<std::string>(input, "base_h5");

  cfg.anchors_total = getScalar<size_t>(anchors, "total", cfg.anchors_total);
  cfg.candidate_min_count = getScalar<int>(anchors, "candidate_min_count", cfg.candidate_min_count);
  {
    const std::vector<int> xyz_bins = getIntList(anchors, "bins_xyz", { cfg.anchor_bins_xyz[0],
                                                                        cfg.anchor_bins_xyz[1],
                                                                        cfg.anchor_bins_xyz[2] });
    if (xyz_bins.size() == 3)
    {
      cfg.anchor_bins_xyz = { xyz_bins[0], xyz_bins[1], xyz_bins[2] };
    }
  }
  {
    const std::vector<int> scr_bins = getIntList(anchors, "bins_scr", { cfg.anchor_bins_scr[0],
                                                                        cfg.anchor_bins_scr[1],
                                                                        cfg.anchor_bins_scr[2] });
    if (scr_bins.size() == 3)
    {
      cfg.anchor_bins_scr = { scr_bins[0], scr_bins[1], scr_bins[2] };
    }
  }
  cfg.anchor_min_quota_xyz = getScalar<size_t>(anchors, "min_quota_xyz", cfg.anchor_min_quota_xyz);
  cfg.anchor_min_quota_scr = getScalar<size_t>(anchors, "min_quota_scr", cfg.anchor_min_quota_scr);
  cfg.anchor_radius_mode = getScalar<std::string>(anchors, "radius_mode", cfg.anchor_radius_mode);
  cfg.anchor_neighbor_mode = getScalar<int>(anchors, "neighbor_mode", cfg.anchor_neighbor_mode);
  YAML::Node weight = anchors["weight"] ? anchors["weight"] : anchors;
  cfg.anchor_eps = getScalar<double>(weight, "epsilon", cfg.anchor_eps);
  cfg.anchor_alpha = getScalar<double>(weight, "alpha", cfg.anchor_alpha);
  cfg.anchor_lambda = getScalar<double>(weight, "lambda", cfg.anchor_lambda);
  cfg.anchor_gamma = getScalar<double>(weight, "gamma", cfg.anchor_gamma);
  cfg.anchor_beta = getScalar<double>(weight, "beta", cfg.anchor_beta);
  cfg.anchor_p = getScalar<double>(weight, "p", cfg.anchor_p);
  cfg.anchor_n0 = getScalar<double>(weight, "n0", cfg.anchor_n0);

  cfg.coarse_min = getScalar<int>(sampling, "coarse_min", cfg.coarse_min);
  cfg.coarse_max = getScalar<int>(sampling, "coarse_max", cfg.coarse_max);
  cfg.knn_k = getScalar<int>(sampling, "knn_k", cfg.knn_k);
  cfg.d_max = getScalar<double>(sampling, "d_max", cfg.d_max);
  cfg.d_max_mode = getScalar<std::string>(sampling, "d_max_mode", cfg.d_max_mode);
  cfg.d_max_factor = getScalar<double>(sampling, "d_max_factor", cfg.d_max_factor);
  cfg.d_max_clip_min = getScalar<double>(sampling, "d_max_clip_min", cfg.d_max_clip_min);
  cfg.d_max_clip_max = getScalar<double>(sampling, "d_max_clip_max", cfg.d_max_clip_max);
  cfg.delta_boundary = getScalar<double>(sampling, "delta_boundary", cfg.delta_boundary);
  cfg.max_boundary_edges = getScalar<size_t>(sampling, "max_boundary_edges", cfg.max_boundary_edges);
  cfg.min_boundary_edges = getScalar<size_t>(sampling, "min_boundary_edges", cfg.min_boundary_edges);
  cfg.shell_deltas = getDoubleList(sampling, "shell_deltas", cfg.shell_deltas);
  cfg.refine_count = getScalar<int>(sampling, "refine_count", cfg.refine_count);
  cfg.refine_sigma = getScalar<double>(sampling, "refine_sigma", cfg.refine_sigma);
  cfg.seeds_per_anchor = getScalar<int>(sampling, "seeds_per_anchor", cfg.seeds_per_anchor);
  cfg.warm_start_mix = getScalar<double>(sampling, "warm_start_mix", cfg.warm_start_mix);
  cfg.scramble_per_anchor = getScalar<bool>(sampling, "scramble_per_anchor", cfg.scramble_per_anchor);
  cfg.sobol_seed = getScalar<uint64_t>(sampling, "sobol_seed", cfg.sobol_seed);
  cfg.q_ref_size = getScalar<int>(sampling, "q_ref_size", cfg.q_ref_size);
  cfg.q_ref_pool = getScalar<int>(sampling, "q_ref_pool", cfg.q_ref_pool);
  cfg.method3_boundary_max = getScalar<int>(sampling, "method3_boundary_max", cfg.method3_boundary_max);
  cfg.method3_boundary_min = getScalar<int>(sampling, "method3_boundary_min", cfg.method3_boundary_min);
  cfg.method3_min_abs_phi = getScalar<double>(sampling, "method3_min_abs_phi", cfg.method3_min_abs_phi);
  cfg.method3_phi_margin = getScalar<double>(sampling, "method3_phi_margin", cfg.method3_phi_margin);

  cfg.ik_timeout = getScalar<double>(ik, "timeout", cfg.ik_timeout);
  cfg.ik_csr_trials = getScalar<int>(ik, "csr_trials", cfg.ik_csr_trials);
  cfg.ik_random_trials = getScalar<int>(ik, "random_trials", cfg.ik_random_trials);
  cfg.bisect_csr_trials = getScalar<int>(ik, "bisect_csr_trials", cfg.bisect_csr_trials);
  cfg.bisect_random_trials = getScalar<int>(ik, "bisect_random_trials", cfg.bisect_random_trials);
  cfg.search_discretization = getScalar<double>(ik, "search_discretization", cfg.search_discretization);

  cfg.threads = getScalar<int>(threads, "count", cfg.threads);

  cfg.debug_max_anchors = getScalar<int>(debug, "max_anchors", cfg.debug_max_anchors);
  cfg.debug_start_anchor = getScalar<int>(debug, "start_anchor", cfg.debug_start_anchor);

  cfg.output_path = getRequiredScalar<std::string>(output, "path");
  cfg.hdf5_chunk = getScalar<size_t>(output, "hdf5_chunk", cfg.hdf5_chunk);
  cfg.write_joint = getScalar<bool>(output, "write_joint", cfg.write_joint);
  cfg.write_method = getScalar<bool>(output, "write_method", cfg.write_method);
  cfg.write_csr = getScalar<bool>(output, "write_csr", cfg.write_csr);
  cfg.flush_per_anchor = getScalar<bool>(output, "flush_per_anchor", cfg.flush_per_anchor);
  cfg.stream_csr = getScalar<bool>(output, "stream_csr", cfg.stream_csr);
  cfg.anchor_only = getScalar<bool>(output, "anchor_only", cfg.anchor_only);

  return cfg;
}

template <typename T>
hid_t h5Type();

template <>
hid_t h5Type<float>()
{
  return H5T_IEEE_F32LE;
}

template <>
hid_t h5Type<double>()
{
  return H5T_IEEE_F64LE;
}

template <>
hid_t h5Type<uint8_t>()
{
  return H5T_STD_U8LE;
}

template <>
hid_t h5Type<uint32_t>()
{
  return H5T_STD_U32LE;
}

template <>
hid_t h5Type<uint64_t>()
{
  return H5T_STD_U64LE;
}

template <typename T>
std::vector<T> readDataset(hid_t file, const std::string& path, std::vector<hsize_t>* dims_out = nullptr)
{
  hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  if (dset < 0)
  {
    throw std::runtime_error("Failed to open dataset: " + path);
  }
  hid_t space = H5Dget_space(dset);
  const int rank = H5Sget_simple_extent_ndims(space);
  std::vector<hsize_t> dims(static_cast<size_t>(rank), 0);
  H5Sget_simple_extent_dims(space, dims.data(), nullptr);
  H5Sclose(space);
  size_t total = 1;
  for (hsize_t d : dims)
  {
    total *= static_cast<size_t>(d);
  }
  std::vector<T> data(total);
  if (H5Dread(dset, h5Type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) < 0)
  {
    H5Dclose(dset);
    throw std::runtime_error("Failed to read dataset: " + path);
  }
  H5Dclose(dset);
  if (dims_out)
  {
    *dims_out = dims;
  }
  return data;
}

std::string readStringDataset(hid_t file, const std::string& path, const std::string& fallback)
{
  hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  if (dset < 0)
  {
    return fallback;
  }
  hid_t dtype = H5Dget_type(dset);
  if (H5Tget_class(dtype) != H5T_STRING)
  {
    H5Tclose(dtype);
    H5Dclose(dset);
    return fallback;
  }
  hid_t memtype = H5Tcopy(H5T_C_S1);
  H5Tset_size(memtype, H5T_VARIABLE);
  char* buffer = nullptr;
  std::string out = fallback;
  if (H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &buffer) >= 0)
  {
    if (buffer)
    {
      out = buffer;
      H5free_memory(buffer);
    }
  }
  H5Tclose(memtype);
  H5Tclose(dtype);
  H5Dclose(dset);
  return out;
}

Eigen::Quaterniond sampleUniformQuaternion(random_numbers::RandomNumberGenerator& rng)
{
  const double u1 = rng.uniformReal(0.0, 1.0);
  const double u2 = rng.uniformReal(0.0, 1.0);
  const double u3 = rng.uniformReal(0.0, 1.0);

  const double sqrt1 = std::sqrt(1.0 - u1);
  const double sqrt2 = std::sqrt(u1);
  const double theta1 = 2.0 * M_PI * u2;
  const double theta2 = 2.0 * M_PI * u3;

  const double x = sqrt1 * std::sin(theta1);
  const double y = sqrt1 * std::cos(theta1);
  const double z = sqrt2 * std::sin(theta2);
  const double w = sqrt2 * std::cos(theta2);

  return Eigen::Quaterniond(w, x, y, z);
}

Eigen::Quaterniond canonicalizeQuat(const Eigen::Quaterniond& q)
{
  Eigen::Quaterniond out = q;
  if (out.w() < 0.0)
  {
    out.coeffs() *= -1.0;
  }
  return out;
}

double geodesicDistance(const Eigen::Quaterniond& a, const Eigen::Quaterniond& b)
{
  double dot = a.dot(b);
  if (dot < 0.0)
  {
    dot = -dot;
  }
  dot = std::min(1.0, std::max(-1.0, dot));
  return 2.0 * std::acos(dot);
}

Eigen::Quaterniond sampleShoemake(double u1, double u2, double u3)
{
  const double sqrt1 = std::sqrt(1.0 - u1);
  const double sqrt2 = std::sqrt(u1);
  const double theta1 = 2.0 * M_PI * u2;
  const double theta2 = 2.0 * M_PI * u3;

  const double x = sqrt1 * std::sin(theta1);
  const double y = sqrt1 * std::cos(theta1);
  const double z = sqrt2 * std::sin(theta2);
  const double w = sqrt2 * std::cos(theta2);
  return Eigen::Quaterniond(w, x, y, z);
}

std::vector<Eigen::Quaterniond> generateSobolQuats(size_t count, uint64_t seed, bool canonicalize)
{
  boost::random::sobol sobol(3);
  if (seed != 0)
  {
    sobol.seed(seed);
  }
  boost::random::uniform_01<double> u01;

  std::vector<Eigen::Quaterniond> quats;
  quats.reserve(count);
  for (size_t i = 0; i < count; ++i)
  {
    const double u1 = u01(sobol);
    const double u2 = u01(sobol);
    const double u3 = u01(sobol);
    Eigen::Quaterniond q = sampleShoemake(u1, u2, u3);
    q.normalize();
    if (canonicalize)
    {
      q = canonicalizeQuat(q);
    }
    quats.push_back(q);
  }
  return quats;
}

struct KNNGraph
{
  int k = 0;
  std::vector<std::vector<int>> neighbors;
  std::vector<std::vector<float>> distances;
  double median_edge = 0.0;
};

struct SampleSet
{
  std::vector<Eigen::Quaterniond> quats;
  KNNGraph knn;
};

KNNGraph buildKNNGraph(const std::vector<Eigen::Quaterniond>& quats, int k, int thread_count)
{
  KNNGraph graph;
  graph.k = k;
  const size_t n = quats.size();
  graph.neighbors.assign(n, std::vector<int>());
  graph.distances.assign(n, std::vector<float>());
  if (n == 0 || k <= 0)
  {
    return graph;
  }

  const int kk = std::min(static_cast<int>(n) - 1, k);
  graph.k = kk;
  graph.neighbors.assign(n, std::vector<int>(static_cast<size_t>(kk), -1));
  graph.distances.assign(n, std::vector<float>(static_cast<size_t>(kk), 0.0f));

#pragma omp parallel for schedule(static) num_threads(thread_count)
  for (int i = 0; i < static_cast<int>(n); ++i)
  {
    std::vector<std::pair<double, int>> heap;
    heap.reserve(static_cast<size_t>(kk));
    const Eigen::Quaterniond& qi = quats[static_cast<size_t>(i)];
    for (int j = 0; j < static_cast<int>(n); ++j)
    {
      if (i == j)
      {
        continue;
      }
      const double d = geodesicDistance(qi, quats[static_cast<size_t>(j)]);
      if (static_cast<int>(heap.size()) < kk)
      {
        heap.emplace_back(d, j);
        std::push_heap(heap.begin(), heap.end(),
                       [](const auto& a, const auto& b) { return a.first < b.first; });
      }
      else if (!heap.empty() && d < heap.front().first)
      {
        std::pop_heap(heap.begin(), heap.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
        heap.back() = { d, j };
        std::push_heap(heap.begin(), heap.end(),
                       [](const auto& a, const auto& b) { return a.first < b.first; });
      }
    }
    std::sort(heap.begin(), heap.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (int t = 0; t < kk; ++t)
    {
      graph.neighbors[static_cast<size_t>(i)][static_cast<size_t>(t)] = heap[static_cast<size_t>(t)].second;
      graph.distances[static_cast<size_t>(i)][static_cast<size_t>(t)] =
          static_cast<float>(heap[static_cast<size_t>(t)].first);
    }
  }

  std::vector<double> all_dist;
  all_dist.reserve(n * static_cast<size_t>(graph.k));
  for (size_t i = 0; i < n; ++i)
  {
    for (int t = 0; t < graph.k; ++t)
    {
      all_dist.push_back(graph.distances[i][static_cast<size_t>(t)]);
    }
  }
  if (!all_dist.empty())
  {
    const size_t mid = all_dist.size() / 2;
    std::nth_element(all_dist.begin(), all_dist.begin() + static_cast<long>(mid), all_dist.end());
    graph.median_edge = all_dist[mid];
  }
  return graph;
}

double resolveDmax(const Config& cfg, const KNNGraph& graph)
{
  if (cfg.d_max_mode == "adaptive")
  {
    const double val = cfg.d_max_factor * graph.median_edge;
    return std::min(cfg.d_max_clip_max, std::max(cfg.d_max_clip_min, val));
  }
  return cfg.d_max;
}

std::vector<float> graphDistance(const KNNGraph& graph, const std::vector<int8_t>& labels, size_t n_used,
                                 double d_max, size_t* boundary_count)
{
  std::vector<float> dist(n_used, std::numeric_limits<float>::infinity());
  if (n_used == 0 || graph.k <= 0)
  {
    return dist;
  }
  std::vector<uint8_t> is_boundary(n_used, 0);
  for (size_t i = 0; i < n_used; ++i)
  {
    for (int t = 0; t < graph.k; ++t)
    {
      const int j = graph.neighbors[i][static_cast<size_t>(t)];
      if (j < 0 || static_cast<size_t>(j) >= n_used)
      {
        continue;
      }
      const float w = graph.distances[i][static_cast<size_t>(t)];
      if (d_max > 0.0 && w > d_max)
      {
        continue;
      }
      if (labels[i] != labels[static_cast<size_t>(j)])
      {
        is_boundary[i] = 1;
        break;
      }
    }
  }

  size_t boundary_nodes = 0;
  for (size_t i = 0; i < n_used; ++i)
  {
    boundary_nodes += is_boundary[i] ? 1 : 0;
  }
  if (boundary_count)
  {
    *boundary_count = boundary_nodes;
  }

  using Node = std::pair<float, int>;
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
  for (size_t i = 0; i < n_used; ++i)
  {
    if (is_boundary[i])
    {
      dist[i] = 0.0f;
      pq.emplace(0.0f, static_cast<int>(i));
    }
  }
  while (!pq.empty())
  {
    const auto [d, i] = pq.top();
    pq.pop();
    if (d > dist[static_cast<size_t>(i)])
    {
      continue;
    }
    for (int t = 0; t < graph.k; ++t)
    {
      const int j = graph.neighbors[static_cast<size_t>(i)][static_cast<size_t>(t)];
      if (j < 0 || static_cast<size_t>(j) >= n_used)
      {
        continue;
      }
      const float w = graph.distances[static_cast<size_t>(i)][static_cast<size_t>(t)];
      const float nd = d + w;
      if (nd < dist[static_cast<size_t>(j)])
      {
        dist[static_cast<size_t>(j)] = nd;
        pq.emplace(nd, j);
      }
    }
  }
  return dist;
}

Eigen::Quaterniond slerpShortest(const Eigen::Quaterniond& a, const Eigen::Quaterniond& b, double t)
{
  Eigen::Quaterniond b_use = b;
  if (a.dot(b) < 0.0)
  {
    b_use.coeffs() *= -1.0;
  }
  return a.slerp(t, b_use);
}

Eigen::Quaterniond expMapSO3(const Eigen::Vector3d& w)
{
  const double theta = w.norm();
  if (theta < 1e-12)
  {
    return Eigen::Quaterniond::Identity();
  }
  const Eigen::Vector3d axis = w / theta;
  const double half = 0.5 * theta;
  const double s = std::sin(half);
  return Eigen::Quaterniond(std::cos(half), axis.x() * s, axis.y() * s, axis.z() * s);
}

const OrientationSample* nearestSample(const std::vector<OrientationSample>& samples, const Eigen::Quaterniond& q)
{
  if (samples.empty())
  {
    return nullptr;
  }
  double best = std::numeric_limits<double>::infinity();
  size_t best_idx = 0;
  for (size_t i = 0; i < samples.size(); ++i)
  {
    const double d = geodesicDistance(samples[i].quat, q);
    if (d < best)
    {
      best = d;
      best_idx = i;
    }
  }
  return &samples[best_idx];
}

int findBin(double value, const std::vector<double>& bins)
{
  if (bins.size() < 2)
  {
    return -1;
  }
  if (value <= bins.front())
  {
    return 0;
  }
  if (value >= bins.back())
  {
    return static_cast<int>(bins.size() - 2);
  }
  for (size_t i = 0; i + 1 < bins.size(); ++i)
  {
    if (value > bins[i] && value <= bins[i + 1])
    {
      return static_cast<int>(i);
    }
  }
  return -1;
}

int binIndex01(double value, int bins)
{
  if (bins <= 0)
  {
    return -1;
  }
  const double v = std::min(1.0, std::max(0.0, value));
  int idx = static_cast<int>(v * static_cast<double>(bins));
  if (idx >= bins)
  {
    idx = bins - 1;
  }
  if (idx < 0)
  {
    idx = 0;
  }
  return idx;
}

std::vector<Seed> selectFarthestSeeds(const std::vector<Seed>& seeds, int max_count)
{
  if (max_count <= 0 || seeds.empty())
  {
    return {};
  }
  if (static_cast<int>(seeds.size()) <= max_count)
  {
    return seeds;
  }

  std::vector<Seed> selected;
  selected.reserve(static_cast<size_t>(max_count));
  selected.push_back(seeds.front());

  std::vector<double> min_dist(seeds.size(), std::numeric_limits<double>::infinity());
  for (size_t i = 0; i < seeds.size(); ++i)
  {
    min_dist[i] = geodesicDistance(seeds[i].quat, selected.front().quat);
  }

  while (static_cast<int>(selected.size()) < max_count)
  {
    size_t best = 0;
    double best_val = -1.0;
    for (size_t i = 0; i < seeds.size(); ++i)
    {
      if (min_dist[i] > best_val)
      {
        best_val = min_dist[i];
        best = i;
      }
    }
    selected.push_back(seeds[best]);
    for (size_t i = 0; i < seeds.size(); ++i)
    {
      const double d = geodesicDistance(seeds[i].quat, seeds[best].quat);
      if (d < min_dist[i])
      {
        min_dist[i] = d;
      }
    }
  }
  return selected;
}

std::vector<Eigen::Quaterniond> selectFarthestQuats(const std::vector<Eigen::Quaterniond>& quats, int max_count)
{
  if (max_count <= 0 || quats.empty())
  {
    return {};
  }
  if (static_cast<int>(quats.size()) <= max_count)
  {
    return quats;
  }
  std::vector<Seed> seeds;
  seeds.reserve(quats.size());
  for (const auto& q : quats)
  {
    Seed s;
    s.quat = q;
    seeds.push_back(s);
  }
  const auto selected = selectFarthestSeeds(seeds, max_count);
  std::vector<Eigen::Quaterniond> out;
  out.reserve(selected.size());
  for (const auto& s : selected)
  {
    out.push_back(s.quat);
  }
  return out;
}

size_t nearestSeedIndex(const std::vector<Seed>& seeds, const Eigen::Quaterniond& quat)
{
  double best = std::numeric_limits<double>::infinity();
  size_t best_idx = 0;
  for (size_t i = 0; i < seeds.size(); ++i)
  {
    const double d = geodesicDistance(seeds[i].quat, quat);
    if (d < best)
    {
      best = d;
      best_idx = i;
    }
  }
  return best_idx;
}

struct ThreadContext
{
  moveit::core::RobotState state;
  collision_detection::CollisionResult result;
  random_numbers::RandomNumberGenerator rng;
  std::vector<double> joint_positions;

  ThreadContext(const moveit::core::RobotModelPtr& model, int seed, size_t joint_count)
    : state(model), rng(seed), joint_positions(joint_count, 0.0)
  {
    state.setToDefaultValues();
  }
};

bool evaluateIK(const Eigen::Vector3d& pos, const Eigen::Quaterniond& quat, const std::string& ee_link,
                const moveit::core::JointModelGroup* jmg, planning_scene::PlanningScenePtr scene,
                collision_detection::CollisionRequest& request, ThreadContext& ctx, const std::vector<Seed>& seeds,
                double warm_start_mix, int csr_trials, int random_trials, double timeout, std::vector<double>& solution)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = pos.x();
  pose.position.y = pos.y();
  pose.position.z = pos.z();
  pose.orientation.w = quat.w();
  pose.orientation.x = quat.x();
  pose.orientation.y = quat.y();
  pose.orientation.z = quat.z();

  auto try_seed = [&](const std::vector<double>* seed) -> bool {
    if (seed)
    {
      ctx.state.setJointGroupPositions(jmg, *seed);
    }
    else
    {
      ctx.state.setToRandomPositions(jmg, ctx.rng);
    }
    const bool ok = ctx.state.setFromIK(jmg, pose, ee_link, timeout);
    if (!ok)
    {
      return false;
    }
    ctx.state.update();
    if (!ctx.state.satisfiesBounds(jmg))
    {
      return false;
    }
    ctx.result.clear();
    scene->checkSelfCollision(request, ctx.result, ctx.state);
    if (ctx.result.collision)
    {
      return false;
    }
    ctx.state.copyJointGroupPositions(jmg, solution);
    return true;
  };

  if (!seeds.empty() && csr_trials > 0)
  {
    for (int trial = 0; trial < csr_trials; ++trial)
    {
      const double u = ctx.rng.uniformReal(0.0, 1.0);
      size_t idx = 0;
      if (u < warm_start_mix)
      {
        idx = nearestSeedIndex(seeds, quat);
      }
      else
      {
        idx = static_cast<size_t>(ctx.rng.uniformInteger(0, static_cast<int>(seeds.size() - 1)));
      }
      if (try_seed(&seeds[idx].joint))
      {
        return true;
      }
    }
  }

  for (int trial = 0; trial < random_trials; ++trial)
  {
    if (try_seed(nullptr))
    {
      return true;
    }
  }
  return false;
}

struct Reservoir
{
  size_t capacity = 0;
  std::vector<std::pair<double, uint32_t>> heap;

  explicit Reservoir(size_t cap = 0) : capacity(cap)
  {
    heap.reserve(capacity);
  }

  void add(double key, uint32_t idx)
  {
    if (capacity == 0)
    {
      return;
    }
    if (heap.size() < capacity)
    {
      heap.emplace_back(key, idx);
      std::push_heap(heap.begin(), heap.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
      return;
    }
    if (!heap.empty() && key < heap.front().first)
    {
      std::pop_heap(heap.begin(), heap.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
      heap.back() = { key, idx };
      std::push_heap(heap.begin(), heap.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    }
  }
};

struct Hdf5Writer
{
  hid_t file = -1;
  hid_t anchors_group = -1;
  hid_t samples_group = -1;
  hid_t csr_group = -1;
  hid_t meta_group = -1;
  hid_t quat_dset = -1;
  hid_t phi_dset = -1;
  hid_t label_dset = -1;
  hid_t joint_dset = -1;
  hid_t method_dset = -1;
  hid_t anchor_start_dset = -1;
  bool write_method = true;
  size_t sample_count = 0;
  size_t joint_count = 0;
  size_t chunk_rows = 0;

  bool open(const std::string& path, size_t chunk, size_t joint_dim, bool enable_method)
  {
    if (chunk == 0)
    {
      chunk = 1;
    }
    file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0)
    {
      return false;
    }
    anchors_group = H5Gcreate2(file, "/anchors", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    samples_group = H5Gcreate2(file, "/samples", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    csr_group = H5Gcreate2(file, "/csr", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    meta_group = H5Gcreate2(file, "/meta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (anchors_group < 0 || samples_group < 0 || csr_group < 0 || meta_group < 0)
    {
      return false;
    }

    chunk_rows = chunk;
    joint_count = joint_dim;
    write_method = enable_method;

    auto create2d = [&](const char* name, hid_t dtype, hsize_t cols) -> hid_t {
      hsize_t dims[2] = { 0, cols };
      hsize_t max_dims[2] = { H5S_UNLIMITED, cols };
      hsize_t chunk_dims[2] = { static_cast<hsize_t>(chunk_rows), cols };
      hid_t space = H5Screate_simple(2, dims, max_dims);
      hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
      H5Pset_chunk(plist, 2, chunk_dims);
      hid_t dset = H5Dcreate2(samples_group, name, dtype, space, H5P_DEFAULT, plist, H5P_DEFAULT);
      H5Pclose(plist);
      H5Sclose(space);
      return dset;
    };

    auto create1d = [&](const char* name, hid_t dtype) -> hid_t {
      hsize_t dims[1] = { 0 };
      hsize_t max_dims[1] = { H5S_UNLIMITED };
      hsize_t chunk_dims[1] = { static_cast<hsize_t>(chunk_rows) };
      hid_t space = H5Screate_simple(1, dims, max_dims);
      hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
      H5Pset_chunk(plist, 1, chunk_dims);
      hid_t dset = H5Dcreate2(samples_group, name, dtype, space, H5P_DEFAULT, plist, H5P_DEFAULT);
      H5Pclose(plist);
      H5Sclose(space);
      return dset;
    };

    quat_dset = create2d("quat", H5T_IEEE_F32LE, 4);
    phi_dset = create1d("phi", H5T_IEEE_F32LE);
    label_dset = create1d("label", H5T_STD_I8LE);
    if (write_method)
    {
      method_dset = create1d("method", H5T_STD_U8LE);
    }
    if (joint_count > 0)
    {
      joint_dset = create2d("joint", H5T_IEEE_F32LE, static_cast<hsize_t>(joint_count));
    }
    return quat_dset >= 0 && phi_dset >= 0 && label_dset >= 0 && (!write_method || method_dset >= 0);
  }

  bool initAnchorStart()
  {
    if (csr_group < 0)
    {
      return false;
    }
    hsize_t dims[1] = { 0 };
    hsize_t max_dims[1] = { H5S_UNLIMITED };
    hsize_t chunk_dims[1] = { static_cast<hsize_t>(chunk_rows) };
    hid_t space = H5Screate_simple(1, dims, max_dims);
    hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(plist, 1, chunk_dims);
    anchor_start_dset = H5Dcreate2(csr_group, "anchor_start", H5T_STD_U64LE, space, H5P_DEFAULT, plist, H5P_DEFAULT);
    H5Pclose(plist);
    H5Sclose(space);
    return anchor_start_dset >= 0;
  }

  bool appendAnchorStart(uint64_t value)
  {
    if (anchor_start_dset < 0)
    {
      return false;
    }
    hid_t space = H5Dget_space(anchor_start_dset);
    hsize_t dims[1] = { 0 };
    H5Sget_simple_extent_dims(space, dims, nullptr);
    H5Sclose(space);

    hsize_t new_dims[1] = { dims[0] + 1 };
    if (H5Dset_extent(anchor_start_dset, new_dims) < 0)
    {
      return false;
    }
    hid_t filespace = H5Dget_space(anchor_start_dset);
    hsize_t start[1] = { dims[0] };
    hsize_t block[1] = { 1 };
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, block, nullptr);
    hid_t memspace = H5Screate_simple(1, block, nullptr);
    const bool ok = H5Dwrite(anchor_start_dset, H5T_STD_U64LE, memspace, filespace, H5P_DEFAULT, &value) >= 0;
    H5Sclose(memspace);
    H5Sclose(filespace);
    return ok;
  }

  void flush()
  {
    if (file >= 0)
    {
      H5Fflush(file, H5F_SCOPE_GLOBAL);
    }
  }

  bool appendSamples(const std::vector<OutputSample>& samples)
  {
    if (samples.empty())
    {
      return true;
    }
    const size_t count = samples.size();
    std::vector<float> quat_buf(count * 4, 0.0f);
    std::vector<float> phi_buf(count, 0.0f);
    std::vector<int8_t> label_buf(count, 0);
    std::vector<uint8_t> method_buf;
    std::vector<float> joint_buf;
    if (write_method)
    {
      method_buf.assign(count, 0);
    }
    if (joint_count > 0)
    {
      joint_buf.resize(count * joint_count, std::numeric_limits<float>::quiet_NaN());
    }

    for (size_t i = 0; i < count; ++i)
    {
      quat_buf[i * 4 + 0] = static_cast<float>(samples[i].quat.x());
      quat_buf[i * 4 + 1] = static_cast<float>(samples[i].quat.y());
      quat_buf[i * 4 + 2] = static_cast<float>(samples[i].quat.z());
      quat_buf[i * 4 + 3] = static_cast<float>(samples[i].quat.w());
      phi_buf[i] = samples[i].phi;
      label_buf[i] = samples[i].label;
      if (write_method)
      {
        method_buf[i] = samples[i].method;
      }
      if (!joint_buf.empty() && samples[i].joint.size() == joint_count)
      {
        const size_t offset = i * joint_count;
        for (size_t j = 0; j < joint_count; ++j)
        {
          joint_buf[offset + j] = samples[i].joint[j];
        }
      }
    }

    auto append2d = [&](hid_t dset, hid_t dtype, const void* data, hsize_t cols) -> bool {
      hid_t space = H5Dget_space(dset);
      hsize_t dims[2] = { 0, 0 };
      H5Sget_simple_extent_dims(space, dims, nullptr);
      H5Sclose(space);

      hsize_t new_dims[2] = { dims[0] + static_cast<hsize_t>(count), cols };
      if (H5Dset_extent(dset, new_dims) < 0)
      {
        return false;
      }
      hid_t filespace = H5Dget_space(dset);
      hsize_t start[2] = { dims[0], 0 };
      hsize_t block[2] = { static_cast<hsize_t>(count), cols };
      H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, block, nullptr);
      hid_t memspace = H5Screate_simple(2, block, nullptr);
      const bool ok = H5Dwrite(dset, dtype, memspace, filespace, H5P_DEFAULT, data) >= 0;
      H5Sclose(memspace);
      H5Sclose(filespace);
      return ok;
    };

    auto append1d = [&](hid_t dset, hid_t dtype, const void* data) -> bool {
      hid_t space = H5Dget_space(dset);
      hsize_t dims[1] = { 0 };
      H5Sget_simple_extent_dims(space, dims, nullptr);
      H5Sclose(space);

      hsize_t new_dims[1] = { dims[0] + static_cast<hsize_t>(count) };
      if (H5Dset_extent(dset, new_dims) < 0)
      {
        return false;
      }
      hid_t filespace = H5Dget_space(dset);
      hsize_t start[1] = { dims[0] };
      hsize_t block[1] = { static_cast<hsize_t>(count) };
      H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, block, nullptr);
      hid_t memspace = H5Screate_simple(1, block, nullptr);
      const bool ok = H5Dwrite(dset, dtype, memspace, filespace, H5P_DEFAULT, data) >= 0;
      H5Sclose(memspace);
      H5Sclose(filespace);
      return ok;
    };

    if (!append2d(quat_dset, H5T_IEEE_F32LE, quat_buf.data(), 4))
    {
      return false;
    }
    if (!append1d(phi_dset, H5T_IEEE_F32LE, phi_buf.data()))
    {
      return false;
    }
    if (!append1d(label_dset, H5T_STD_I8LE, label_buf.data()))
    {
      return false;
    }
    if (write_method)
    {
      if (!append1d(method_dset, H5T_STD_U8LE, method_buf.data()))
      {
        return false;
      }
    }
    if (joint_dset >= 0 && !joint_buf.empty())
    {
      if (!append2d(joint_dset, H5T_IEEE_F32LE, joint_buf.data(), static_cast<hsize_t>(joint_count)))
      {
        return false;
      }
    }

    sample_count += count;
    return true;
  }

  bool writeArray(hid_t group, const std::string& name, hid_t dtype, const std::vector<hsize_t>& dims,
                  const void* data)
  {
    if (group < 0)
    {
      return false;
    }
    hid_t space = H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
    hid_t dset = H5Dcreate2(group, name.c_str(), dtype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0)
    {
      H5Sclose(space);
      return false;
    }
    const bool ok = H5Dwrite(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) >= 0;
    H5Dclose(dset);
    H5Sclose(space);
    return ok;
  }

  bool writeScalar(hid_t group, const std::string& name, hid_t dtype, const void* data)
  {
    hid_t space = H5Screate(H5S_SCALAR);
    if (space < 0)
    {
      return false;
    }
    hid_t dset = H5Dcreate2(group, name.c_str(), dtype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0)
    {
      H5Sclose(space);
      return false;
    }
    const bool ok = H5Dwrite(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) >= 0;
    H5Dclose(dset);
    H5Sclose(space);
    return ok;
  }

  bool writeScalar(hid_t group, const std::string& name, hid_t dtype, double value)
  {
    return writeScalar(group, name, dtype, &value);
  }

  bool writeString(hid_t group, const std::string& name, const std::string& value)
  {
    if (group < 0)
    {
      return false;
    }
    hid_t dtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(dtype, H5T_VARIABLE);
    hsize_t dims[1] = { 1 };
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dset = H5Dcreate2(group, name.c_str(), dtype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const char* data = value.c_str();
    const bool ok = dset >= 0 && H5Dwrite(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data) >= 0;
    if (dset >= 0)
    {
      H5Dclose(dset);
    }
    H5Sclose(space);
    H5Tclose(dtype);
    return ok;
  }

  void close()
  {
    if (quat_dset >= 0)
    {
      H5Dclose(quat_dset);
    }
    if (phi_dset >= 0)
    {
      H5Dclose(phi_dset);
    }
    if (label_dset >= 0)
    {
      H5Dclose(label_dset);
    }
    if (joint_dset >= 0)
    {
      H5Dclose(joint_dset);
    }
    if (method_dset >= 0)
    {
      H5Dclose(method_dset);
    }
    if (anchors_group >= 0)
    {
      H5Gclose(anchors_group);
    }
    if (samples_group >= 0)
    {
      H5Gclose(samples_group);
    }
    if (anchor_start_dset >= 0)
    {
      H5Dclose(anchor_start_dset);
      anchor_start_dset = -1;
    }
    if (csr_group >= 0)
    {
      H5Gclose(csr_group);
    }
    if (meta_group >= 0)
    {
      H5Gclose(meta_group);
    }
    if (file >= 0)
    {
      H5Fclose(file);
    }
  }
};

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  Options options;
  if (!parseArgs(argc, argv, options))
  {
    printUsage();
    rclcpp::shutdown();
    return 2;
  }

  auto node = rclcpp::Node::make_shared("orientation_dataset_cli");
  auto logger = node->get_logger();

  Config cfg;
  try
  {
    cfg = loadConfig(options.config_path);
  }
  catch (const std::exception& ex)
  {
    RCLCPP_ERROR(logger, "Failed to load config: %s", ex.what());
    rclcpp::shutdown();
    return 1;
  }

  int thread_count = cfg.threads;
#ifdef _OPENMP
  if (thread_count <= 0)
  {
    thread_count = omp_get_max_threads();
  }
  omp_set_num_threads(thread_count);
#else
  thread_count = 1;
#endif

  RCLCPP_INFO(logger, "Input HDF5: %s", cfg.base_h5_path.c_str());
  RCLCPP_INFO(logger, "Output HDF5: %s", cfg.output_path.c_str());
  RCLCPP_INFO(logger, "Threads: %d", thread_count);

  hid_t base_file = H5Fopen(cfg.base_h5_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (base_file < 0)
  {
    RCLCPP_ERROR(logger, "Failed to open base HDF5: %s", cfg.base_h5_path.c_str());
    rclcpp::shutdown();
    return 1;
  }

  std::vector<hsize_t> dims_dims;
  std::vector<uint64_t> dims_vec = readDataset<uint64_t>(base_file, "/grid/dims", &dims_dims);
  std::vector<double> origin_vec = readDataset<double>(base_file, "/grid/origin");
  std::vector<double> voxel_vec = readDataset<double>(base_file, "/grid/voxel_size");
  if (dims_vec.size() != 3 || origin_vec.size() != 3 || voxel_vec.empty())
  {
    RCLCPP_ERROR(logger, "Invalid grid metadata in base HDF5");
    H5Fclose(base_file);
    rclcpp::shutdown();
    return 1;
  }

  Grid grid;
  grid.nx = static_cast<size_t>(dims_vec[0]);
  grid.ny = static_cast<size_t>(dims_vec[1]);
  grid.nz = static_cast<size_t>(dims_vec[2]);
  grid.origin = Eigen::Vector3d(origin_vec[0], origin_vec[1], origin_vec[2]);
  grid.voxel_size = voxel_vec[0];

  const size_t grid_size = grid.size();

  RCLCPP_INFO(logger, "Grid: %zu x %zu x %zu voxel=%.4f", grid.nx, grid.ny, grid.nz, grid.voxel_size);

  std::vector<uint8_t> label;
  try
  {
    label = readDataset<uint8_t>(base_file, "/grid/label");
  }
  catch (const std::exception&)
  {
    label = readDataset<uint8_t>(base_file, "/grid/occupancy");
  }
  std::vector<float> sdf = readDataset<float>(base_file, "/grid/sdf");
  std::vector<float> coverage;
  bool has_coverage = true;
  try
  {
    coverage = readDataset<float>(base_file, "/grid/orientation_coverage");
  }
  catch (const std::exception&)
  {
    has_coverage = false;
    coverage.assign(grid_size, 0.0f);
  }
  std::vector<uint32_t> counts = readDataset<uint32_t>(base_file, "/grid/voxel_counts");
  std::vector<uint64_t> voxel_start = readDataset<uint64_t>(base_file, "/csr/voxel_start");
  std::string index_dtype = readStringDataset(base_file, "/csr/index_dtype", "uint32");
  bool sample_index_is_u64 = false;
  std::vector<uint64_t> sample_index_u64;
  std::vector<uint32_t> sample_index_u32;
  if (index_dtype == "uint64")
  {
    sample_index_is_u64 = true;
    sample_index_u64 = readDataset<uint64_t>(base_file, "/csr/sample_index");
  }
  else
  {
    if (index_dtype != "uint32")
    {
      RCLCPP_WARN(logger, "Unknown index_dtype '%s', fallback to uint32", index_dtype.c_str());
    }
    sample_index_u32 = readDataset<uint32_t>(base_file, "/csr/sample_index");
  }
  const size_t sample_index_size =
      sample_index_is_u64 ? sample_index_u64.size() : sample_index_u32.size();

  if (label.size() != grid_size || sdf.size() != grid_size || coverage.size() != grid_size ||
      counts.size() != grid_size)
  {
    RCLCPP_ERROR(logger, "Grid dataset size mismatch");
    H5Fclose(base_file);
    rclcpp::shutdown();
    return 1;
  }

  const int bins_x = std::max(1, cfg.anchor_bins_xyz[0]);
  const int bins_y = std::max(1, cfg.anchor_bins_xyz[1]);
  const int bins_z = std::max(1, cfg.anchor_bins_xyz[2]);
  const int bins_s = std::max(1, cfg.anchor_bins_scr[0]);
  const int bins_c = std::max(1, cfg.anchor_bins_scr[1]);
  const int bins_r = std::max(1, cfg.anchor_bins_scr[2]);
  const size_t xyz_bin_count = static_cast<size_t>(bins_x * bins_y * bins_z);
  const size_t scr_bin_count = static_cast<size_t>(bins_s * bins_c * bins_r);
  if (xyz_bin_count == 0 || scr_bin_count == 0)
  {
    RCLCPP_ERROR(logger, "Invalid anchor bin configuration");
    H5Fclose(base_file);
    rclcpp::shutdown();
    return 1;
  }

  std::vector<float> g_values(grid_size, 0.0f);
  const bool use_l2_radius = (cfg.anchor_radius_mode == "l2");

  size_t inside_count = 0;
  size_t candidate_count = 0;
  const size_t max_quantile_samples = 2000000;
  std::vector<double> s_samples;
  std::vector<double> n_samples;
  s_samples.reserve(std::min(max_quantile_samples, grid_size));
  n_samples.reserve(std::min(max_quantile_samples, grid_size));
  random_numbers::RandomNumberGenerator quant_rng(1337);
  size_t seen_inside = 0;

  for (size_t x = 0; x < grid.nx; ++x)
  {
    for (size_t y = 0; y < grid.ny; ++y)
    {
      for (size_t z = 0; z < grid.nz; ++z)
      {
        const size_t idx = grid.index(x, y, z);
        if (label[idx] != 1)
        {
          continue;
        }
        ++inside_count;
        if (counts[idx] >= static_cast<uint32_t>(std::max(1, cfg.candidate_min_count)))
        {
          ++candidate_count;
        }
        ++seen_inside;
        const double s_val = std::fabs(static_cast<double>(sdf[idx]));
        const double n_val = static_cast<double>(counts[idx]);
        if (s_samples.size() < max_quantile_samples)
        {
          s_samples.push_back(s_val);
          n_samples.push_back(n_val);
        }
        else
        {
          const uint64_t j = static_cast<uint64_t>(quant_rng.uniformInteger(0, static_cast<int>(seen_inside - 1)));
          if (j < max_quantile_samples)
          {
            s_samples[static_cast<size_t>(j)] = s_val;
            n_samples[static_cast<size_t>(j)] = n_val;
          }
        }
      }
    }
  }

  const double s_ref = std::max(1e-9, percentile(s_samples, 0.90));
  const double n_ref = std::max(1e-9, percentile(n_samples, 0.90));

  const Eigen::Vector3d half_extent =
      0.5 * grid.voxel_size * Eigen::Vector3d(static_cast<double>(grid.nx), static_cast<double>(grid.ny),
                                              static_cast<double>(grid.nz));
  const Eigen::Vector3d aabb_center = grid.origin + half_extent;
  const double r_max_l2 = std::max(1e-9, half_extent.norm());
  const double r_max_linf =
      std::max({ std::fabs(half_extent.x()), std::fabs(half_extent.y()), std::fabs(half_extent.z()), 1e-9 });

#pragma omp parallel for schedule(static)
  for (size_t x = 0; x < grid.nx; ++x)
  {
    for (size_t y = 0; y < grid.ny; ++y)
    {
      for (size_t z = 0; z < grid.nz; ++z)
      {
        const size_t idx = grid.index(x, y, z);
        if (label[idx] != 1)
        {
          continue;
        }
        const float c_v = has_coverage ? coverage[idx] : 0.0f;
        float g_v = 0.0f;
        auto try_neighbor = [&](int nx, int ny, int nz) {
          if (nx < 0 || ny < 0 || nz < 0 || nx >= static_cast<int>(grid.nx) ||
              ny >= static_cast<int>(grid.ny) || nz >= static_cast<int>(grid.nz))
          {
            return;
          }
          const size_t nidx =
              grid.index(static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz));
          if (label[nidx] != 1)
          {
            return;
          }
          const float diff = std::fabs(c_v - coverage[nidx]);
          if (diff > g_v)
          {
            g_v = diff;
          }
        };
        if (cfg.anchor_neighbor_mode == 6)
        {
          try_neighbor(static_cast<int>(x) + 1, static_cast<int>(y), static_cast<int>(z));
          try_neighbor(static_cast<int>(x) - 1, static_cast<int>(y), static_cast<int>(z));
          try_neighbor(static_cast<int>(x), static_cast<int>(y) + 1, static_cast<int>(z));
          try_neighbor(static_cast<int>(x), static_cast<int>(y) - 1, static_cast<int>(z));
          try_neighbor(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z) + 1);
          try_neighbor(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z) - 1);
        }
        g_values[idx] = g_v;
      }
    }
  }

  if (candidate_count == 0)
  {
    RCLCPP_ERROR(logger, "No candidate anchors found (label==1 and n_v>=min).");
    H5Fclose(base_file);
    rclcpp::shutdown();
    return 1;
  }

  const size_t total_target = std::min(cfg.anchors_total, candidate_count);
  RCLCPP_INFO(logger, "Anchor target=%zu candidate=%zu bins_xyz=%d,%d,%d bins_scr=%d,%d,%d", total_target,
              candidate_count, bins_x, bins_y, bins_z, bins_s, bins_c, bins_r);

  auto compute_weight = [&](size_t x, size_t y, size_t z, size_t idx, double& s_val, double& s_hat, double& c_hat,
                            double& r_hat, int& bin_xyz, int& bin_scr, double& weight) -> bool {
    if (label[idx] != 1)
    {
      return false;
    }
    if (counts[idx] < static_cast<uint32_t>(std::max(1, cfg.candidate_min_count)))
    {
      return false;
    }
    s_val = std::fabs(static_cast<double>(sdf[idx]));
    s_hat = std::min(1.0, s_val / s_ref);
    c_hat = std::min(1.0, std::max(0.0, static_cast<double>(has_coverage ? coverage[idx] : 0.0f)));
    const double n_hat =
        n_ref > 0.0 ? std::sqrt(std::min(1.0, static_cast<double>(counts[idx]) / n_ref)) : 1.0;
    const Eigen::Vector3d p = grid.center(x, y, z);
    const Eigen::Vector3d d = p - aabb_center;
    const double r_l2 = d.norm() / r_max_l2;
    const double r_linf = std::max({ std::fabs(d.x()), std::fabs(d.y()), std::fabs(d.z()) }) / r_max_linf;
    r_hat = std::min(1.0, use_l2_radius ? r_l2 : r_linf);
    const double r_term = 1.0 + cfg.anchor_beta * std::pow(r_hat, cfg.anchor_p);
    weight = std::pow(1.0 / (s_hat + cfg.anchor_eps), cfg.anchor_alpha) *
             (cfg.anchor_lambda + 4.0 * c_hat * (1.0 - c_hat)) *
             (1.0 + cfg.anchor_gamma * static_cast<double>(g_values[idx])) * n_hat * r_term;
    if (weight <= 0.0)
    {
      return false;
    }

    const double xn =
        (static_cast<double>(x) + 0.5) / std::max(1.0, static_cast<double>(grid.nx));
    const double yn =
        (static_cast<double>(y) + 0.5) / std::max(1.0, static_cast<double>(grid.ny));
    const double zn =
        (static_cast<double>(z) + 0.5) / std::max(1.0, static_cast<double>(grid.nz));
    const int bx = binIndex01(xn, bins_x);
    const int by = binIndex01(yn, bins_y);
    const int bz = binIndex01(zn, bins_z);
    if (bx < 0 || by < 0 || bz < 0)
    {
      return false;
    }
    bin_xyz = (bx * bins_y + by) * bins_z + bz;

    const int bs = binIndex01(s_hat, bins_s);
    const int bc = binIndex01(c_hat, bins_c);
    const int br = binIndex01(r_hat, bins_r);
    if (bs < 0 || bc < 0 || br < 0)
    {
      return false;
    }
    bin_scr = (bs * bins_c + bc) * bins_r + br;
    return true;
  };

  std::vector<std::vector<Reservoir>> thread_xyz(static_cast<size_t>(thread_count),
                                                  std::vector<Reservoir>(xyz_bin_count));
  std::vector<std::vector<Reservoir>> thread_scr(static_cast<size_t>(thread_count),
                                                  std::vector<Reservoir>(scr_bin_count));
  for (int t = 0; t < thread_count; ++t)
  {
    for (size_t b = 0; b < xyz_bin_count; ++b)
    {
      thread_xyz[static_cast<size_t>(t)][b] = Reservoir(cfg.anchor_min_quota_xyz);
    }
    for (size_t b = 0; b < scr_bin_count; ++b)
    {
      thread_scr[static_cast<size_t>(t)][b] = Reservoir(cfg.anchor_min_quota_scr);
    }
  }

#pragma omp parallel
  {
    const int tid = 0
#ifdef _OPENMP
                    + omp_get_thread_num()
#endif
        ;
    random_numbers::RandomNumberGenerator rng(1234 + tid * 97);

#pragma omp for schedule(static)
    for (size_t x = 0; x < grid.nx; ++x)
    {
      for (size_t y = 0; y < grid.ny; ++y)
      {
        for (size_t z = 0; z < grid.nz; ++z)
        {
          const size_t idx = grid.index(x, y, z);
          double s_val = 0.0;
          double s_hat = 0.0;
          double c_hat = 0.0;
          double r_hat = 0.0;
          double weight = 0.0;
          int bin_xyz = -1;
          int bin_scr = -1;
          if (!compute_weight(x, y, z, idx, s_val, s_hat, c_hat, r_hat, bin_xyz, bin_scr, weight))
          {
            continue;
          }
          if (weight <= 0.0)
          {
            continue;
          }
          if (cfg.anchor_min_quota_xyz > 0)
          {
            const double u = rng.uniformReal(0.0, 1.0);
            const double key = -std::log(std::max(u, 1e-12)) / weight;
            thread_xyz[static_cast<size_t>(tid)][static_cast<size_t>(bin_xyz)].add(key, static_cast<uint32_t>(idx));
          }
          if (cfg.anchor_min_quota_scr > 0)
          {
            const double u = rng.uniformReal(0.0, 1.0);
            const double key = -std::log(std::max(u, 1e-12)) / weight;
            thread_scr[static_cast<size_t>(tid)][static_cast<size_t>(bin_scr)].add(key, static_cast<uint32_t>(idx));
          }
        }
      }
    }
  }

  std::unordered_set<uint32_t> selected_ids;
  auto merge_bins = [&](const std::vector<std::vector<Reservoir>>& bins, size_t quota, size_t bin_count) {
    if (quota == 0)
    {
      return;
    }
    for (size_t b = 0; b < bin_count; ++b)
    {
      std::vector<std::pair<double, uint32_t>> all_keys;
      for (int t = 0; t < thread_count; ++t)
      {
        const auto& heap = bins[static_cast<size_t>(t)][b].heap;
        all_keys.insert(all_keys.end(), heap.begin(), heap.end());
      }
      if (all_keys.empty())
      {
        continue;
      }
      const size_t k = std::min(quota, all_keys.size());
      if (k < all_keys.size())
      {
        std::nth_element(all_keys.begin(), all_keys.begin() + static_cast<long>(k), all_keys.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
      }
      for (size_t i = 0; i < k; ++i)
      {
        selected_ids.insert(all_keys[i].second);
      }
    }
  };

  merge_bins(thread_xyz, cfg.anchor_min_quota_xyz, xyz_bin_count);
  merge_bins(thread_scr, cfg.anchor_min_quota_scr, scr_bin_count);

  if (selected_ids.size() > total_target)
  {
    RCLCPP_WARN(logger, "Anchor quotas exceed target (%zu > %zu). Trimming by weight.", selected_ids.size(),
                total_target);
  }

  if (selected_ids.size() > total_target)
  {
    std::vector<std::pair<double, uint32_t>> weighted;
    weighted.reserve(selected_ids.size());
    for (uint32_t idx : selected_ids)
    {
      const size_t x = idx / (grid.ny * grid.nz);
      const size_t rem = idx % (grid.ny * grid.nz);
      const size_t y = rem / grid.nz;
      const size_t z = rem % grid.nz;
      double s_val = 0.0;
      double s_hat = 0.0;
      double c_hat = 0.0;
      double r_hat = 0.0;
      double weight = 0.0;
      int bin_xyz = -1;
      int bin_scr = -1;
      if (!compute_weight(x, y, z, idx, s_val, s_hat, c_hat, r_hat, bin_xyz, bin_scr, weight))
      {
        continue;
      }
      weighted.emplace_back(weight, idx);
    }
    std::sort(weighted.begin(), weighted.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    selected_ids.clear();
    for (size_t i = 0; i < std::min(total_target, weighted.size()); ++i)
    {
      selected_ids.insert(weighted[i].second);
    }
  }

  if (selected_ids.size() < total_target)
  {
    const size_t remaining = total_target - selected_ids.size();
    std::vector<Reservoir> thread_global(static_cast<size_t>(thread_count), Reservoir(remaining));

#pragma omp parallel
    {
      const int tid = 0
#ifdef _OPENMP
                      + omp_get_thread_num()
#endif
          ;
      random_numbers::RandomNumberGenerator rng(4321 + tid * 53);

#pragma omp for schedule(static)
      for (size_t x = 0; x < grid.nx; ++x)
      {
        for (size_t y = 0; y < grid.ny; ++y)
        {
          for (size_t z = 0; z < grid.nz; ++z)
          {
            const size_t idx = grid.index(x, y, z);
            if (selected_ids.find(static_cast<uint32_t>(idx)) != selected_ids.end())
            {
              continue;
            }
            double s_val = 0.0;
            double s_hat = 0.0;
            double c_hat = 0.0;
            double r_hat = 0.0;
            double weight = 0.0;
            int bin_xyz = -1;
            int bin_scr = -1;
            if (!compute_weight(x, y, z, idx, s_val, s_hat, c_hat, r_hat, bin_xyz, bin_scr, weight))
            {
              continue;
            }
            const double u = rng.uniformReal(0.0, 1.0);
            const double key = -std::log(std::max(u, 1e-12)) / weight;
            thread_global[static_cast<size_t>(tid)].add(key, static_cast<uint32_t>(idx));
          }
        }
      }
    }

    std::vector<std::pair<double, uint32_t>> all_keys;
    for (int t = 0; t < thread_count; ++t)
    {
      const auto& heap = thread_global[static_cast<size_t>(t)].heap;
      all_keys.insert(all_keys.end(), heap.begin(), heap.end());
    }
    if (!all_keys.empty())
    {
      const size_t k = std::min(remaining, all_keys.size());
      if (k < all_keys.size())
      {
        std::nth_element(all_keys.begin(), all_keys.begin() + static_cast<long>(k), all_keys.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
      }
      for (size_t i = 0; i < k; ++i)
      {
        selected_ids.insert(all_keys[i].second);
      }
    }
  }

  if (selected_ids.empty())
  {
    RCLCPP_ERROR(logger, "No anchors selected.");
    H5Fclose(base_file);
    rclcpp::shutdown();
    return 1;
  }

  std::vector<Anchor> anchors;
  anchors.reserve(selected_ids.size());
  for (uint32_t idx : selected_ids)
  {
    const size_t x = idx / (grid.ny * grid.nz);
    const size_t rem = idx % (grid.ny * grid.nz);
    const size_t y = rem / grid.nz;
    const size_t z = rem % grid.nz;
    const Eigen::Vector3d p = grid.center(x, y, z);
    double s_val = 0.0;
    double s_hat = 0.0;
    double c_hat = 0.0;
    double r_hat = 0.0;
    double weight = 0.0;
    int bin_xyz = -1;
    int bin_scr = -1;
    if (!compute_weight(x, y, z, idx, s_val, s_hat, c_hat, r_hat, bin_xyz, bin_scr, weight))
    {
      continue;
    }
    Anchor a;
    a.idx = idx;
    a.s_v = static_cast<float>(s_val);
    a.c_v = static_cast<float>(c_hat);
    a.g_v = g_values[idx];
    a.n_seed = counts[idx];
    a.pos = Eigen::Vector3f(static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z()));
    a.weight = weight;
    anchors.push_back(a);
  }

  std::sort(anchors.begin(), anchors.end(), [](const Anchor& a, const Anchor& b) { return a.weight > b.weight; });

  const size_t total_anchors = anchors.size();
  size_t start_anchor = 0;
  if (cfg.debug_start_anchor > 0)
  {
    start_anchor = std::min(static_cast<size_t>(cfg.debug_start_anchor), total_anchors);
  }
  size_t process_count = total_anchors - start_anchor;
  if (cfg.debug_max_anchors > 0)
  {
    process_count = std::min(process_count, static_cast<size_t>(cfg.debug_max_anchors));
  }
  std::vector<uint8_t> anchor_selected(total_anchors, 0);
  for (size_t i = start_anchor; i < start_anchor + process_count; ++i)
  {
    anchor_selected[i] = 1;
  }

  RCLCPP_INFO(logger, "Anchors selected: %zu (processing %zu, start=%zu)", total_anchors, process_count,
              start_anchor);

  reachability_cli::RobotContext context;
  try
  {
    context = reachability_cli::loadRobotFromFiles(cfg.urdf_path, cfg.srdf_path);
  }
  catch (const std::exception& ex)
  {
    RCLCPP_ERROR(logger, "%s", ex.what());
    H5Fclose(base_file);
    rclcpp::shutdown();
    return 1;
  }

  const auto& robot_model = context.model;
  const auto& scene = context.scene;
  const auto* jmg = robot_model->getJointModelGroup(cfg.group_name);
  if (!jmg)
  {
    RCLCPP_ERROR(logger, "JointModelGroup '%s' not found", cfg.group_name.c_str());
    H5Fclose(base_file);
    rclcpp::shutdown();
    return 1;
  }

  const size_t joint_count = jmg->getVariableCount();
  if (!cfg.joint_names.empty() && cfg.joint_names != jmg->getVariableNames())
  {
    RCLCPP_ERROR(logger, "Joint order mismatch for group '%s'", cfg.group_name.c_str());
    H5Fclose(base_file);
    rclcpp::shutdown();
    return 1;
  }

  auto solver_allocator = [node, robot_model, cfg](const moveit::core::JointModelGroup* group)
      -> kinematics::KinematicsBasePtr {
    auto solver = std::make_shared<kdl_kinematics_plugin::KDLKinematicsPlugin>();
    const std::vector<std::string> tips{ cfg.ee_link };
    if (!solver->initialize(node, *robot_model, group->getName(), cfg.base_link, tips, cfg.search_discretization))
    {
      RCLCPP_ERROR(node->get_logger(), "Failed to initialize KDL solver for group '%s'", group->getName().c_str());
      return kinematics::KinematicsBasePtr();
    }
    return kinematics::KinematicsBasePtr(solver);
  };

  const_cast<moveit::core::JointModelGroup*>(jmg)->setSolverAllocators(solver_allocator);
  const_cast<moveit::core::JointModelGroup*>(jmg)->setDefaultIKTimeout(cfg.ik_timeout);

  collision_detection::CollisionRequest request;
  request.group_name = cfg.group_name;
  request.contacts = false;
  request.max_contacts = 0;

  std::vector<ThreadContext> thread_contexts;
  thread_contexts.reserve(static_cast<size_t>(thread_count));
  for (int i = 0; i < thread_count; ++i)
  {
    thread_contexts.emplace_back(robot_model, 2024 + i * 101, joint_count);
  }

  const int coarse_min = std::max(1, cfg.coarse_min);
  const int coarse_max = std::max(coarse_min, cfg.coarse_max);
  SampleSet precomp_min;
  SampleSet precomp_max;
  if (!cfg.scramble_per_anchor)
  {
    RCLCPP_INFO(logger, "Precompute Sobol quats: N=%d (seed=%llu)", coarse_max,
                static_cast<unsigned long long>(cfg.sobol_seed));
    precomp_max.quats = generateSobolQuats(static_cast<size_t>(coarse_max), cfg.sobol_seed, false);
    precomp_max.knn = buildKNNGraph(precomp_max.quats, cfg.knn_k, thread_count);
    if (coarse_min != coarse_max)
    {
      precomp_min.quats.assign(precomp_max.quats.begin(),
                               precomp_max.quats.begin() + static_cast<long>(coarse_min));
      precomp_min.knn = buildKNNGraph(precomp_min.quats, cfg.knn_k, thread_count);
    }
    else
    {
      precomp_min = precomp_max;
    }
  }
  else
  {
    RCLCPP_WARN(logger, "scramble_per_anchor=true; Sobol samples and kNN will be built per anchor.");
  }

  std::vector<Eigen::Quaterniond> q_ref;
  if (cfg.q_ref_size > 0)
  {
    const int q_pool = std::max(cfg.q_ref_pool, cfg.q_ref_size);
    const uint64_t q_seed = cfg.sobol_seed + 0x9e3779b97f4a7c15ULL;
    RCLCPP_INFO(logger, "Q_ref: pool=%d size=%d (seed=%llu)", q_pool, cfg.q_ref_size,
                static_cast<unsigned long long>(q_seed));
    const auto q_pool_quats = generateSobolQuats(static_cast<size_t>(q_pool), q_seed, true);
    q_ref = selectFarthestQuats(q_pool_quats, cfg.q_ref_size);
  }

  Hdf5Writer writer;
  if (!writer.open(cfg.output_path, cfg.hdf5_chunk, cfg.write_joint ? joint_count : 0, cfg.write_method))
  {
    RCLCPP_ERROR(logger, "Failed to create output HDF5: %s", cfg.output_path.c_str());
    H5Fclose(base_file);
    rclcpp::shutdown();
    return 1;
  }
  if (cfg.write_csr && cfg.stream_csr)
  {
    if (!writer.initAnchorStart())
    {
      RCLCPP_WARN(logger, "Failed to init streaming /csr/anchor_start; fall back to end-write.");
      cfg.stream_csr = false;
    }
  }
  writer.writeString(writer.meta_group, "base_h5_path", cfg.base_h5_path);
  writer.writeString(writer.meta_group, "config_path", options.config_path);
  {
    std::ifstream config_in(options.config_path);
    if (config_in)
    {
      std::stringstream buffer;
      buffer << config_in.rdbuf();
      writer.writeString(writer.meta_group, "config_yaml", buffer.str());
    }
  }
  writer.writeScalar(writer.meta_group, "s_ref_p90", H5T_IEEE_F64LE, s_ref);
  writer.writeScalar(writer.meta_group, "n_ref_p90", H5T_IEEE_F64LE, n_ref);
  writer.writeScalar(writer.meta_group, "r_max_l2", H5T_IEEE_F64LE, r_max_l2);
  writer.writeScalar(writer.meta_group, "r_max_linf", H5T_IEEE_F64LE, r_max_linf);
  if (!q_ref.empty())
  {
    std::vector<float> q_ref_buf;
    q_ref_buf.reserve(q_ref.size() * 4);
    for (const auto& q : q_ref)
    {
      const Eigen::Quaterniond qc = canonicalizeQuat(q);
      q_ref_buf.push_back(static_cast<float>(qc.x()));
      q_ref_buf.push_back(static_cast<float>(qc.y()));
      q_ref_buf.push_back(static_cast<float>(qc.z()));
      q_ref_buf.push_back(static_cast<float>(qc.w()));
    }
    writer.writeArray(writer.meta_group, "q_ref", H5T_IEEE_F32LE,
                      { static_cast<hsize_t>(q_ref.size()), 4 }, q_ref_buf.data());
    writer.writeScalar(writer.meta_group, "q_ref_size", H5T_STD_I32LE, cfg.q_ref_size);
    writer.writeScalar(writer.meta_group, "q_ref_pool", H5T_STD_I32LE, cfg.q_ref_pool);
  }

  std::vector<uint64_t> anchor_ids;
  std::vector<float> anchor_pos;
  std::vector<float> anchor_s;
  std::vector<float> anchor_c;
  std::vector<float> anchor_g;
  std::vector<uint32_t> anchor_n;

  for (const auto& a : anchors)
  {
    anchor_ids.push_back(static_cast<uint64_t>(a.idx));
    anchor_pos.push_back(a.pos.x());
    anchor_pos.push_back(a.pos.y());
    anchor_pos.push_back(a.pos.z());
    anchor_s.push_back(a.s_v);
    anchor_c.push_back(a.c_v);
    anchor_g.push_back(a.g_v);
    anchor_n.push_back(a.n_seed);
  }

  writer.writeArray(writer.anchors_group, "voxel_id", H5T_STD_U64LE, { anchor_ids.size() }, anchor_ids.data());
  writer.writeArray(writer.anchors_group, "pos", H5T_IEEE_F32LE, { anchor_ids.size(), 3 }, anchor_pos.data());
  writer.writeArray(writer.anchors_group, "s_v", H5T_IEEE_F32LE, { anchor_ids.size() }, anchor_s.data());
  writer.writeArray(writer.anchors_group, "c_v", H5T_IEEE_F32LE, { anchor_ids.size() }, anchor_c.data());
  writer.writeArray(writer.anchors_group, "g_v", H5T_IEEE_F32LE, { anchor_ids.size() }, anchor_g.data());
  writer.writeArray(writer.anchors_group, "n_seed", H5T_STD_U32LE, { anchor_ids.size() }, anchor_n.data());
  writer.writeArray(writer.anchors_group, "selected", H5T_STD_U8LE, { anchor_selected.size() }, anchor_selected.data());

  if (cfg.anchor_only)
  {
    writer.writeScalar(writer.meta_group, "anchor_only", H5T_STD_I8LE, 1);
    RCLCPP_INFO(logger, "Anchor-only mode enabled. Wrote anchors to %s", cfg.output_path.c_str());
    H5Fclose(base_file);
    rclcpp::shutdown();
    return 0;
  }

  std::vector<uint64_t> anchor_start;
  anchor_start.reserve(anchors.size() + 1);
  anchor_start.push_back(0);
  if (cfg.write_csr && cfg.stream_csr)
  {
    writer.appendAnchorStart(0);
    if (cfg.flush_per_anchor)
    {
      writer.flush();
    }
  }

  auto readSeeds = [&](uint64_t voxel_idx) -> std::vector<Seed> {
    std::vector<Seed> seeds;
    if (voxel_idx + 1 >= voxel_start.size())
    {
      return seeds;
    }
    const uint64_t start = voxel_start[static_cast<size_t>(voxel_idx)];
    const uint64_t end = voxel_start[static_cast<size_t>(voxel_idx + 1)];
    if (end <= start)
    {
      return seeds;
    }
    if (start >= sample_index_size)
    {
      return seeds;
    }
    const size_t end_cap = static_cast<size_t>(std::min<uint64_t>(end, sample_index_size));
    const size_t safe_count = end_cap - static_cast<size_t>(start);
    std::vector<uint64_t> ids(safe_count, 0);
    for (size_t i = 0; i < safe_count; ++i)
    {
      const size_t idx = static_cast<size_t>(start) + i;
      ids[i] = sample_index_is_u64 ? sample_index_u64[idx] : sample_index_u32[idx];
    }

    hid_t quat_dset = H5Dopen2(base_file, "/samples/quat", H5P_DEFAULT);
    hid_t joint_dset = H5Dopen2(base_file, "/samples/joint", H5P_DEFAULT);
    if (quat_dset < 0 || joint_dset < 0)
    {
      if (quat_dset >= 0)
      {
        H5Dclose(quat_dset);
      }
      if (joint_dset >= 0)
      {
        H5Dclose(joint_dset);
      }
      return seeds;
    }

    seeds.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i)
    {
      const hsize_t row = static_cast<hsize_t>(ids[i]);
      hsize_t start_row[2] = { row, 0 };
      hsize_t count_row[2] = { 1, 4 };
      std::array<float, 4> qbuf{};
      hid_t qspace = H5Dget_space(quat_dset);
      H5Sselect_hyperslab(qspace, H5S_SELECT_SET, start_row, nullptr, count_row, nullptr);
      hid_t qmem = H5Screate_simple(2, count_row, nullptr);
      H5Dread(quat_dset, H5T_IEEE_F32LE, qmem, qspace, H5P_DEFAULT, qbuf.data());
      H5Sclose(qmem);
      H5Sclose(qspace);

      hsize_t jcount[2] = { 1, static_cast<hsize_t>(joint_count) };
      std::vector<float> jbuf(joint_count, 0.0f);
      hid_t jspace = H5Dget_space(joint_dset);
      H5Sselect_hyperslab(jspace, H5S_SELECT_SET, start_row, nullptr, jcount, nullptr);
      hid_t jmem = H5Screate_simple(2, jcount, nullptr);
      H5Dread(joint_dset, H5T_IEEE_F32LE, jmem, jspace, H5P_DEFAULT, jbuf.data());
      H5Sclose(jmem);
      H5Sclose(jspace);

      Seed seed;
      seed.quat = Eigen::Quaterniond(qbuf[3], qbuf[0], qbuf[1], qbuf[2]);
      seed.quat.normalize();
      seed.joint.assign(jbuf.begin(), jbuf.end());
      seeds.push_back(seed);
    }

    H5Dclose(quat_dset);
    H5Dclose(joint_dset);
    return seeds;
  };

  auto generateRandomSeeds = [&](size_t count, int seed) -> std::vector<Seed> {
    std::vector<Seed> out;
    if (count == 0)
    {
      return out;
    }
    moveit::core::RobotState state(robot_model);
    state.setToDefaultValues();
    random_numbers::RandomNumberGenerator rng(seed);
    out.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
      state.setToRandomPositions(jmg, rng);
      state.update();
      const auto& tf = state.getGlobalLinkTransform(cfg.ee_link);
      Eigen::Quaterniond q(tf.rotation());
      q.normalize();
      Seed seed_entry;
      seed_entry.quat = q;
      seed_entry.joint.resize(joint_count, 0.0);
      state.copyJointGroupPositions(jmg, seed_entry.joint);
      out.push_back(seed_entry);
    }
    return out;
  };

  auto build_edges = [&](const KNNGraph& graph, const std::vector<int8_t>& labels, size_t n_used, double d_max) {
    struct Edge
    {
      int i;
      int j;
      float dist;
    };
    std::vector<Edge> edges;
    edges.reserve(n_used * static_cast<size_t>(graph.k));
    std::unordered_set<uint64_t> seen;
    for (size_t i = 0; i < n_used; ++i)
    {
      for (int t = 0; t < graph.k; ++t)
      {
        const int j = graph.neighbors[i][static_cast<size_t>(t)];
        if (j < 0 || static_cast<size_t>(j) >= n_used)
        {
          continue;
        }
        if (labels[i] == labels[static_cast<size_t>(j)])
        {
          continue;
        }
        const float dist = graph.distances[i][static_cast<size_t>(t)];
        if (dist > d_max)
        {
          continue;
        }
        const uint32_t a = static_cast<uint32_t>(std::min(static_cast<int>(i), j));
        const uint32_t b = static_cast<uint32_t>(std::max(static_cast<int>(i), j));
        const uint64_t key = (static_cast<uint64_t>(a) << 32) | b;
        if (seen.insert(key).second)
        {
          edges.push_back({ static_cast<int>(a), static_cast<int>(b), dist });
        }
      }
    }
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) { return a.dist < b.dist; });
    if (edges.size() > cfg.max_boundary_edges)
    {
      edges.resize(cfg.max_boundary_edges);
    }
    return edges;
  };

  for (size_t ai = 0; ai < anchors.size(); ++ai)
  {
    if (ai < anchor_selected.size() && anchor_selected[ai] == 0)
    {
      anchor_start.push_back(writer.sample_count);
      if (cfg.write_csr && cfg.stream_csr)
      {
        writer.appendAnchorStart(writer.sample_count);
        if (cfg.flush_per_anchor)
        {
          writer.flush();
        }
      }
      continue;
    }

    const Anchor& anchor = anchors[ai];
    const Eigen::Vector3d pos(anchor.pos.x(), anchor.pos.y(), anchor.pos.z());

    std::vector<Seed> seeds = readSeeds(anchor.idx);
    if (seeds.size() < static_cast<size_t>(cfg.seeds_per_anchor))
    {
      const size_t missing = static_cast<size_t>(cfg.seeds_per_anchor) - seeds.size();
      auto extra = generateRandomSeeds(missing, 9001 + static_cast<int>(ai) * 17);
      seeds.insert(seeds.end(), extra.begin(), extra.end());
    }
    seeds = selectFarthestSeeds(seeds, cfg.seeds_per_anchor);
    if (seeds.empty())
    {
      RCLCPP_WARN(logger, "Anchor %zu has no seeds; skipping.", ai);
      anchor_start.push_back(writer.sample_count);
      continue;
    }

    SampleSet local_min;
    SampleSet local_max;
    const SampleSet* set_min = &precomp_min;
    const SampleSet* set_max = &precomp_max;
    if (cfg.scramble_per_anchor)
    {
      const uint64_t seed = cfg.sobol_seed + static_cast<uint64_t>(ai) + 1;
      local_max.quats = generateSobolQuats(static_cast<size_t>(coarse_max), seed, false);
      local_max.knn = buildKNNGraph(local_max.quats, cfg.knn_k, thread_count);
      if (coarse_min != coarse_max)
      {
        local_min.quats.assign(local_max.quats.begin(), local_max.quats.begin() + static_cast<long>(coarse_min));
        local_min.knn = buildKNNGraph(local_min.quats, cfg.knn_k, thread_count);
      }
      else
      {
        local_min = local_max;
      }
      set_min = &local_min;
      set_max = &local_max;
    }

    size_t n_used = static_cast<size_t>(coarse_min);
    const SampleSet* samples = set_min;

    std::vector<OrientationSample> all_samples(n_used);
    std::vector<int8_t> labels(n_used, 0);

    auto eval_range = [&](size_t start, size_t end) {
#pragma omp parallel for schedule(static)
      for (int i = static_cast<int>(start); i < static_cast<int>(end); ++i)
      {
        const int tid = 0
#ifdef _OPENMP
                        + omp_get_thread_num()
#endif
            ;
        auto& ctx = thread_contexts[static_cast<size_t>(tid)];
        std::vector<double> solution(joint_count, 0.0);
        const bool ok = evaluateIK(pos, samples->quats[static_cast<size_t>(i)], cfg.ee_link, jmg, scene, request, ctx,
                                   seeds, cfg.warm_start_mix, cfg.ik_csr_trials, cfg.ik_random_trials, cfg.ik_timeout,
                                   solution);
        all_samples[static_cast<size_t>(i)].quat = samples->quats[static_cast<size_t>(i)];
        all_samples[static_cast<size_t>(i)].reachable = ok;
        labels[static_cast<size_t>(i)] = ok ? 1 : -1;
        if (ok)
        {
          all_samples[static_cast<size_t>(i)].joint = solution;
        }
        else
        {
          all_samples[static_cast<size_t>(i)].joint.clear();
        }
      }
    };

    eval_range(0, n_used);

    double d_max_used = resolveDmax(cfg, samples->knn);
    auto edges = build_edges(samples->knn, labels, n_used, d_max_used);

    if (edges.size() < cfg.min_boundary_edges && coarse_max > coarse_min)
    {
      RCLCPP_INFO(logger, "Anchor %zu: boundary edges %zu < %zu, extending to %d samples", ai, edges.size(),
                  cfg.min_boundary_edges, coarse_max);
      samples = set_max;
      const size_t old_used = n_used;
      n_used = static_cast<size_t>(coarse_max);
      all_samples.resize(n_used);
      labels.resize(n_used, 0);
      eval_range(old_used, n_used);
      d_max_used = resolveDmax(cfg, samples->knn);
      edges = build_edges(samples->knn, labels, n_used, d_max_used);
    }

    std::vector<OrientationSample> pos_samples;
    std::vector<OrientationSample> neg_samples;
    pos_samples.reserve(n_used);
    neg_samples.reserve(n_used);
    for (size_t i = 0; i < n_used; ++i)
    {
      if (labels[i] == 1)
      {
        pos_samples.push_back(all_samples[i]);
      }
      else
      {
        neg_samples.push_back(all_samples[i]);
      }
    }

    if (pos_samples.empty() || neg_samples.empty())
    {
      RCLCPP_WARN(logger, "Anchor %zu has insufficient pos/neg samples (pos=%zu neg=%zu).", ai, pos_samples.size(),
                  neg_samples.size());
    }

    std::vector<OutputSample> output_samples;
    std::vector<Eigen::Quaterniond> boundary_quats;

    if (!edges.empty())
    {
      const size_t log_step = std::max<size_t>(1, edges.size() / 5);
      std::atomic<size_t> processed{ 0 };
      std::atomic<size_t> next_log{ log_step };
      std::vector<std::vector<OutputSample>> output_local(static_cast<size_t>(thread_count));
      std::vector<std::vector<Eigen::Quaterniond>> boundary_local(static_cast<size_t>(thread_count));
      std::vector<random_numbers::RandomNumberGenerator> refine_rngs;
      refine_rngs.reserve(static_cast<size_t>(thread_count));
      for (int t = 0; t < thread_count; ++t)
      {
        refine_rngs.emplace_back(7001 + static_cast<int>(ai) * 101 + t * 11);
      }

#pragma omp parallel for schedule(dynamic)
      for (int e = 0; e < static_cast<int>(edges.size()); ++e)
      {
        const int tid = 0
#ifdef _OPENMP
                        + omp_get_thread_num()
#endif
            ;
        auto& ctx = thread_contexts[static_cast<size_t>(tid)];
        auto& rng = refine_rngs[static_cast<size_t>(tid)];
        auto& out = output_local[static_cast<size_t>(tid)];
        auto& bq = boundary_local[static_cast<size_t>(tid)];
        const auto& edge = edges[static_cast<size_t>(e)];

        const int idx_a = edge.i;
        const int idx_b = edge.j;
        const bool a_pos = labels[static_cast<size_t>(idx_a)] == 1;
        const int pos_idx = a_pos ? idx_a : idx_b;
        const int neg_idx = a_pos ? idx_b : idx_a;

        const Eigen::Quaterniond q_pos_far = all_samples[static_cast<size_t>(pos_idx)].quat;
        const Eigen::Quaterniond q_neg_far = all_samples[static_cast<size_t>(neg_idx)].quat;
        Eigen::Quaterniond q_pos = q_pos_far;
        Eigen::Quaterniond q_neg = q_neg_far;
        std::vector<double> q_pos_joint = all_samples[static_cast<size_t>(pos_idx)].joint;
        std::vector<double> solution(joint_count, 0.0);

        while (geodesicDistance(q_pos, q_neg) > cfg.delta_boundary)
        {
          const Eigen::Quaterniond q_mid = slerpShortest(q_pos, q_neg, 0.5);
          const bool ok = evaluateIK(pos, q_mid, cfg.ee_link, jmg, scene, request, ctx, seeds, cfg.warm_start_mix,
                                     cfg.bisect_csr_trials, cfg.bisect_random_trials, cfg.ik_timeout, solution);
          if (ok)
          {
            q_pos = q_mid;
            q_pos_joint = solution;
          }
          else
          {
            q_neg = q_mid;
          }
        }

        const Eigen::Quaterniond q_star = q_pos;
        bq.push_back(q_star);
        OutputSample boundary;
        boundary.quat = canonicalizeQuat(q_star);
        boundary.phi = 0.0f;
        boundary.label = 1;
        boundary.method = 2;
        out.push_back(boundary);

        const double d_pos = geodesicDistance(q_star, q_pos_far);
        const double d_neg = geodesicDistance(q_star, q_neg_far);
        for (double delta : cfg.shell_deltas)
        {
          if (d_pos > 1e-9 && delta <= d_pos)
          {
            const double t = delta / d_pos;
            Eigen::Quaterniond q_in = slerpShortest(q_star, q_pos_far, t);
            OutputSample s;
            s.quat = canonicalizeQuat(q_in);
            s.phi = static_cast<float>(delta);
            s.label = 1;
            s.method = 0;
            s.joint.assign(q_pos_joint.begin(), q_pos_joint.end());
            out.push_back(s);
          }
          if (d_neg > 1e-9 && delta <= d_neg)
          {
            const double t = delta / d_neg;
            Eigen::Quaterniond q_out = slerpShortest(q_star, q_neg_far, t);
            OutputSample s;
            s.quat = canonicalizeQuat(q_out);
            s.phi = static_cast<float>(-delta);
            s.label = -1;
            s.method = 1;
            out.push_back(s);
          }
        }

        if (cfg.refine_count > 0)
        {
          for (int r = 0; r < cfg.refine_count; ++r)
          {
            const Eigen::Vector3d noise(rng.gaussian(0.0, cfg.refine_sigma), rng.gaussian(0.0, cfg.refine_sigma),
                                        rng.gaussian(0.0, cfg.refine_sigma));
            Eigen::Quaterniond dq = expMapSO3(noise);
            Eigen::Quaterniond q_pert = dq * q_star;
            q_pert.normalize();

            std::vector<double> pert_solution(joint_count, 0.0);
            const bool pert_ok =
                evaluateIK(pos, q_pert, cfg.ee_link, jmg, scene, request, ctx, seeds, cfg.warm_start_mix,
                           cfg.bisect_csr_trials, cfg.bisect_random_trials, cfg.ik_timeout, pert_solution);

            const OrientationSample* opp =
                pert_ok ? nearestSample(neg_samples, q_pert) : nearestSample(pos_samples, q_pert);
            if (!opp)
            {
              continue;
            }
            if (geodesicDistance(q_pert, opp->quat) > d_max_used)
            {
              continue;
            }

            Eigen::Quaterniond pos_end = pert_ok ? q_pert : opp->quat;
            Eigen::Quaterniond neg_end = pert_ok ? opp->quat : q_pert;
            std::vector<double> pos_joint = pert_ok ? pert_solution : opp->joint;

            Eigen::Quaterniond pos_local = pos_end;
            Eigen::Quaterniond neg_local = neg_end;
            std::vector<double> local_solution = pos_joint;

            while (geodesicDistance(pos_local, neg_local) > cfg.delta_boundary)
            {
              const Eigen::Quaterniond q_mid = slerpShortest(pos_local, neg_local, 0.5);
              std::vector<double> mid_solution(joint_count, 0.0);
              const bool ok =
                  evaluateIK(pos, q_mid, cfg.ee_link, jmg, scene, request, ctx, seeds, cfg.warm_start_mix,
                             cfg.bisect_csr_trials, cfg.bisect_random_trials, cfg.ik_timeout, mid_solution);
              if (ok)
              {
                pos_local = q_mid;
                local_solution = mid_solution;
              }
              else
              {
                neg_local = q_mid;
              }
            }

            const Eigen::Quaterniond q_star_ref = pos_local;
            bq.push_back(q_star_ref);
            OutputSample b;
            b.quat = canonicalizeQuat(q_star_ref);
            b.phi = 0.0f;
            b.label = 1;
            b.method = 2;
            out.push_back(b);

            const double d_pos_ref = geodesicDistance(q_star_ref, pos_end);
            const double d_neg_ref = geodesicDistance(q_star_ref, neg_end);
            for (double delta : cfg.shell_deltas)
            {
              if (d_pos_ref > 1e-9 && delta <= d_pos_ref)
              {
                const double t = delta / d_pos_ref;
                Eigen::Quaterniond q_in = slerpShortest(q_star_ref, pos_end, t);
                OutputSample s;
                s.quat = canonicalizeQuat(q_in);
                s.phi = static_cast<float>(delta);
                s.label = 1;
                s.method = 0;
                s.joint.assign(local_solution.begin(), local_solution.end());
                out.push_back(s);
              }
              if (d_neg_ref > 1e-9 && delta <= d_neg_ref)
              {
                const double t = delta / d_neg_ref;
                Eigen::Quaterniond q_out = slerpShortest(q_star_ref, neg_end, t);
                OutputSample s;
                s.quat = canonicalizeQuat(q_out);
                s.phi = static_cast<float>(-delta);
                s.label = -1;
                s.method = 1;
                out.push_back(s);
              }
            }
          }
        }

        const size_t done = processed.fetch_add(1) + 1;
        size_t target = next_log.load();
        if (done >= target)
        {
          if (next_log.compare_exchange_strong(target, target + log_step))
          {
            RCLCPP_INFO(logger, "Anchor %zu phase2/3 %zu/%zu", ai, done, edges.size());
          }
        }
      }

      for (int t = 0; t < thread_count; ++t)
      {
        boundary_quats.insert(boundary_quats.end(), boundary_local[static_cast<size_t>(t)].begin(),
                              boundary_local[static_cast<size_t>(t)].end());
        output_samples.insert(output_samples.end(), output_local[static_cast<size_t>(t)].begin(),
                              output_local[static_cast<size_t>(t)].end());
      }
    }
    else
    {
      RCLCPP_WARN(logger, "Anchor %zu has no boundary edges after filtering.", ai);
    }

    double method3_min_phi = cfg.method3_min_abs_phi;
    if (method3_min_phi <= 0.0)
    {
      double max_shell = 0.0;
      for (double d : cfg.shell_deltas)
      {
        max_shell = std::max(max_shell, d);
      }
      method3_min_phi = max_shell + cfg.delta_boundary + cfg.method3_phi_margin;
    }

    size_t coarse_added = 0;
    std::vector<Eigen::Quaterniond> boundary_use = boundary_quats;
    if (!boundary_use.empty() && cfg.method3_boundary_max > 0 &&
        static_cast<int>(boundary_use.size()) > cfg.method3_boundary_max)
    {
      const int target = std::max(cfg.method3_boundary_min, cfg.method3_boundary_max);
      std::vector<size_t> idx(boundary_use.size());
      std::iota(idx.begin(), idx.end(), 0);
      std::mt19937 rng(static_cast<uint32_t>(cfg.sobol_seed + 1337 + ai * 1013));
      std::shuffle(idx.begin(), idx.end(), rng);
      std::vector<Eigen::Quaterniond> reduced;
      reduced.reserve(static_cast<size_t>(target));
      for (int i = 0; i < target && i < static_cast<int>(idx.size()); ++i)
      {
        reduced.push_back(boundary_use[idx[static_cast<size_t>(i)]]);
      }
      boundary_use.swap(reduced);
    }
    if (boundary_use.empty())
    {
      RCLCPP_WARN(logger, "Anchor %zu has no boundary points; skip method=3 samples.", ai);
    }
    else
    {
      std::vector<Eigen::Vector4f> bvec;
      bvec.reserve(boundary_use.size());
      for (const auto& q : boundary_use)
      {
        Eigen::Quaterniond qc = q;
        qc.normalize();
        bvec.push_back(qc.coeffs().cast<float>());
      }

      std::vector<std::vector<OutputSample>> local_out(static_cast<size_t>(thread_count));
#pragma omp parallel for schedule(static)
      for (int i = 0; i < static_cast<int>(n_used); ++i)
      {
        const int tid = 0
#ifdef _OPENMP
                        + omp_get_thread_num()
#endif
            ;
        const auto& qi = all_samples[static_cast<size_t>(i)].quat;
        Eigen::Quaterniond qn = qi.normalized();
        Eigen::Vector4f qv = qn.coeffs().cast<float>();
        float best = -1.0f;
        for (const auto& bq : bvec)
        {
          const float d = std::fabs(qv.dot(bq));
          if (d > best)
          {
            best = d;
          }
        }
        if (best < 0.0f)
        {
          continue;
        }
        best = std::min(1.0f, std::max(-1.0f, best));
        const float dist = 2.0f * std::acos(best);
        if (method3_min_phi > 0.0 && dist <= static_cast<float>(method3_min_phi))
        {
          continue;
        }
        OutputSample o;
        o.quat = canonicalizeQuat(qn);
        o.label = labels[static_cast<size_t>(i)] == 1 ? 1 : -1;
        o.method = 3;
        o.phi = o.label > 0 ? dist : -dist;
        if (o.label > 0)
        {
          o.joint.assign(all_samples[static_cast<size_t>(i)].joint.begin(),
                         all_samples[static_cast<size_t>(i)].joint.end());
        }
        local_out[static_cast<size_t>(tid)].push_back(std::move(o));
      }
      for (auto& vec : local_out)
      {
        coarse_added += vec.size();
        output_samples.insert(output_samples.end(), vec.begin(), vec.end());
      }
    }

    if (!writer.appendSamples(output_samples))
    {
      RCLCPP_ERROR(logger, "Failed to append samples for anchor %zu", ai);
      writer.close();
      H5Fclose(base_file);
      rclcpp::shutdown();
      return 1;
    }

    anchor_start.push_back(writer.sample_count);
    if (cfg.write_csr && cfg.stream_csr)
    {
      writer.appendAnchorStart(writer.sample_count);
      if (cfg.flush_per_anchor)
      {
        writer.flush();
      }
    }
    RCLCPP_INFO(logger, "Anchor %zu done: boundary=%zu coarse=%zu total=%zu", ai, boundary_quats.size(),
                coarse_added, output_samples.size());
  }

  if (cfg.write_csr && !cfg.stream_csr)
  {
    writer.writeArray(writer.csr_group, "anchor_start", H5T_STD_U64LE,
                      { static_cast<hsize_t>(anchor_start.size()) }, anchor_start.data());
  }
  if (cfg.write_csr)
  {
    std::vector<uint64_t> sample_ids(writer.sample_count, 0);
    std::iota(sample_ids.begin(), sample_ids.end(), 0);
    writer.writeArray(writer.csr_group, "sample_index", H5T_STD_U64LE,
                      { static_cast<hsize_t>(sample_ids.size()) }, sample_ids.data());
  }

  writer.close();
  H5Fclose(base_file);
  RCLCPP_INFO(logger, "Done. Output: %s", cfg.output_path.c_str());
  rclcpp::shutdown();
  return 0;
}
