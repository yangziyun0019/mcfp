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

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <unordered_map>
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

struct QuatGridKey
{
  int x = 0;
  int y = 0;
  int z = 0;
  int w = 0;

  bool operator==(const QuatGridKey& other) const
  {
    return x == other.x && y == other.y && z == other.z && w == other.w;
  }
};

struct QuatGridKeyHash
{
  size_t operator()(const QuatGridKey& key) const
  {
    size_t h = 1469598103934665603ULL;
    auto mix = [&](int v) {
      h ^= static_cast<size_t>(static_cast<uint32_t>(v));
      h *= 1099511628211ULL;
    };
    mix(key.x);
    mix(key.y);
    mix(key.z);
    mix(key.w);
    return h;
  }
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

  size_t anchors_total = 1024;
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

  std::string so3_grid = "hopf";
  int so3_base_cells = 576;
  int so3_split_factor = 8;
  int so3_max_depth = 3;
  double so3_leaf_radius = 0.04;
  double so3_stencil_radius_ratio = 0.5;
  double so3_refine_band = 0.08;
  double delta_boundary = 0.01;
  int boundary_refine_steps = 2;
  int max_boundary_brackets = 0;
  std::vector<double> shell_deltas{ 0.02, 0.04 };
  double phi_full = M_PI;
  int full_reachable_validation_samples = 1024;
  uint64_t full_reachable_validation_seed = 20260429ULL;
  int seeds_per_anchor = 32;
  double warm_start_mix = 0.7;
  int q_ref_size = 64;
  int q_ref_pool = 4096;
  uint64_t q_ref_seed = 20260428ULL;
  uint64_t anchor_quantile_seed = 2026042801ULL;
  uint64_t anchor_quota_seed_base = 2026042802ULL;
  uint64_t anchor_quota_seed_stride = 97ULL;
  uint64_t anchor_global_seed_base = 2026042803ULL;
  uint64_t anchor_global_seed_stride = 53ULL;
  uint64_t thread_seed_base = 2026042804ULL;
  uint64_t thread_seed_stride = 101ULL;
  uint64_t fallback_seed_base = 2026042805ULL;
  uint64_t fallback_seed_stride = 17ULL;

  double ik_timeout_coarse = 0.005;
  int ik_csr_trials_coarse = 2;
  int ik_random_trials_coarse = 1;
  double ik_timeout_refine = 0.01;
  int ik_csr_trials_refine = 3;
  int ik_random_trials_refine = 2;
  double ik_timeout_bisect = 0.02;
  int ik_csr_trials_bisect = 4;
  int ik_random_trials_bisect = 4;
  double search_discretization = 0.005;

  int threads = 0;

  int debug_max_anchors = 1;
  int debug_start_anchor = 0;
  std::vector<int> debug_anchor_indices;

  std::string output_path;
  size_t hdf5_chunk = 100000;
  bool write_joint = true;
  bool write_method = true;
  bool write_cells = true;
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

struct OutputSample
{
  Eigen::Quaterniond quat;
  float phi = 0.0f;
  int8_t label = 0;
  uint8_t method = 0;
  std::vector<float> joint;
};

enum class So3CellState : uint8_t
{
  kInside = 0,
  kOutside = 1,
  kMixed = 2,
  kUnknown = 3
};

struct So3Cell
{
  int parent = -1;
  uint8_t level = 0;
  double radius = 0.0;
  Eigen::Quaterniond center = Eigen::Quaterniond::Identity();
  So3CellState state = So3CellState::kUnknown;
  std::array<Eigen::Quaterniond, 7> stencil_quat{};
  std::array<int8_t, 7> stencil_label{};
  std::array<std::vector<double>, 7> stencil_joint{};
  std::vector<double> center_joint;
  bool has_center_joint = false;
  float phi_graph = std::numeric_limits<float>::quiet_NaN();
};

struct CellOutput
{
  Eigen::Quaterniond quat;
  uint8_t level = 0;
  uint8_t state = 3;
  float phi_graph = std::numeric_limits<float>::quiet_NaN();
};

struct BoundaryBracket
{
  Eigen::Quaterniond q_pos = Eigen::Quaterniond::Identity();
  Eigen::Quaterniond q_neg = Eigen::Quaterniond::Identity();
  std::vector<double> pos_joint;
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

  cfg.so3_grid = getScalar<std::string>(sampling, "so3_grid", cfg.so3_grid);
  cfg.so3_base_cells = getScalar<int>(sampling, "so3_base_cells", cfg.so3_base_cells);
  cfg.so3_split_factor = getScalar<int>(sampling, "so3_split_factor", cfg.so3_split_factor);
  cfg.so3_max_depth = getScalar<int>(sampling, "so3_max_depth", cfg.so3_max_depth);
  cfg.so3_leaf_radius = getScalar<double>(sampling, "so3_leaf_radius", cfg.so3_leaf_radius);
  cfg.so3_stencil_radius_ratio =
      getScalar<double>(sampling, "so3_stencil_radius_ratio", cfg.so3_stencil_radius_ratio);
  cfg.so3_refine_band = getScalar<double>(sampling, "so3_refine_band", cfg.so3_refine_band);
  cfg.delta_boundary = getScalar<double>(sampling, "delta_boundary", cfg.delta_boundary);
  cfg.boundary_refine_steps = getScalar<int>(sampling, "boundary_refine_steps", cfg.boundary_refine_steps);
  cfg.max_boundary_brackets = getScalar<int>(sampling, "max_boundary_brackets", cfg.max_boundary_brackets);
  cfg.shell_deltas = getDoubleList(sampling, "shell_deltas", cfg.shell_deltas);
  cfg.phi_full = getScalar<double>(sampling, "phi_full", cfg.phi_full);
  cfg.full_reachable_validation_samples =
      getScalar<int>(sampling, "full_reachable_validation_samples", cfg.full_reachable_validation_samples);
  cfg.full_reachable_validation_seed =
      getScalar<uint64_t>(sampling, "full_reachable_validation_seed", cfg.full_reachable_validation_seed);
  cfg.seeds_per_anchor = getScalar<int>(sampling, "seeds_per_anchor", cfg.seeds_per_anchor);
  cfg.warm_start_mix = getScalar<double>(sampling, "warm_start_mix", cfg.warm_start_mix);
  cfg.q_ref_size = getScalar<int>(sampling, "q_ref_size", cfg.q_ref_size);
  cfg.q_ref_pool = getScalar<int>(sampling, "q_ref_pool", cfg.q_ref_pool);
  cfg.q_ref_seed = getScalar<uint64_t>(sampling, "q_ref_seed", cfg.q_ref_seed);
  cfg.anchor_quantile_seed = getScalar<uint64_t>(sampling, "anchor_quantile_seed", cfg.anchor_quantile_seed);
  cfg.anchor_quota_seed_base = getScalar<uint64_t>(sampling, "anchor_quota_seed_base", cfg.anchor_quota_seed_base);
  cfg.anchor_quota_seed_stride =
      getScalar<uint64_t>(sampling, "anchor_quota_seed_stride", cfg.anchor_quota_seed_stride);
  cfg.anchor_global_seed_base = getScalar<uint64_t>(sampling, "anchor_global_seed_base", cfg.anchor_global_seed_base);
  cfg.anchor_global_seed_stride =
      getScalar<uint64_t>(sampling, "anchor_global_seed_stride", cfg.anchor_global_seed_stride);
  cfg.thread_seed_base = getScalar<uint64_t>(sampling, "thread_seed_base", cfg.thread_seed_base);
  cfg.thread_seed_stride = getScalar<uint64_t>(sampling, "thread_seed_stride", cfg.thread_seed_stride);
  cfg.fallback_seed_base = getScalar<uint64_t>(sampling, "fallback_seed_base", cfg.fallback_seed_base);
  cfg.fallback_seed_stride = getScalar<uint64_t>(sampling, "fallback_seed_stride", cfg.fallback_seed_stride);

  cfg.ik_timeout_coarse = getScalar<double>(ik, "timeout_coarse", cfg.ik_timeout_coarse);
  cfg.ik_csr_trials_coarse = getScalar<int>(ik, "csr_trials_coarse", cfg.ik_csr_trials_coarse);
  cfg.ik_random_trials_coarse = getScalar<int>(ik, "random_trials_coarse", cfg.ik_random_trials_coarse);
  cfg.ik_timeout_refine = getScalar<double>(ik, "timeout_refine", cfg.ik_timeout_refine);
  cfg.ik_csr_trials_refine = getScalar<int>(ik, "csr_trials_refine", cfg.ik_csr_trials_refine);
  cfg.ik_random_trials_refine = getScalar<int>(ik, "random_trials_refine", cfg.ik_random_trials_refine);
  cfg.ik_timeout_bisect = getScalar<double>(ik, "timeout_bisect", cfg.ik_timeout_bisect);
  cfg.ik_csr_trials_bisect = getScalar<int>(ik, "csr_trials_bisect", cfg.ik_csr_trials_bisect);
  cfg.ik_random_trials_bisect = getScalar<int>(ik, "random_trials_bisect", cfg.ik_random_trials_bisect);
  cfg.search_discretization = getScalar<double>(ik, "search_discretization", cfg.search_discretization);

  cfg.threads = getScalar<int>(threads, "count", cfg.threads);

  cfg.debug_max_anchors = getScalar<int>(debug, "max_anchors", cfg.debug_max_anchors);
  cfg.debug_start_anchor = getScalar<int>(debug, "start_anchor", cfg.debug_start_anchor);
  cfg.debug_anchor_indices = getIntList(debug, "anchor_indices", {});

  cfg.output_path = getRequiredScalar<std::string>(output, "path");
  cfg.hdf5_chunk = getScalar<size_t>(output, "hdf5_chunk", cfg.hdf5_chunk);
  cfg.write_joint = getScalar<bool>(output, "write_joint", cfg.write_joint);
  cfg.write_method = getScalar<bool>(output, "write_method", cfg.write_method);
  cfg.write_cells = getScalar<bool>(output, "write_cells", cfg.write_cells);
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

bool datasetExists(hid_t file, const std::string& path)
{
  H5E_auto2_t old_func = nullptr;
  void* old_client_data = nullptr;
  H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
  H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);
  if (dset < 0)
  {
    return false;
  }
  H5Dclose(dset);
  return true;
}

int nearestCoverageBin(const Eigen::Quaterniond& quat, const std::vector<std::array<float, 4>>& bins)
{
  if (bins.empty())
  {
    return -1;
  }
  const Eigen::Quaterniond q = quat.normalized();
  const float qx = static_cast<float>(q.x());
  const float qy = static_cast<float>(q.y());
  const float qz = static_cast<float>(q.z());
  const float qw = static_cast<float>(q.w());
  float best = -1.0f;
  int best_idx = 0;
  for (int i = 0; i < static_cast<int>(bins.size()); ++i)
  {
    const auto& b = bins[static_cast<size_t>(i)];
    float dot = qx * b[0] + qy * b[1] + qz * b[2] + qw * b[3];
    if (dot < 0.0f)
    {
      dot = -dot;
    }
    if (dot > best)
    {
      best = dot;
      best_idx = i;
    }
  }
  return best_idx;
}

void setBit(std::vector<uint8_t>& bits, size_t row, size_t cols, size_t col)
{
  if (cols == 0 || row >= bits.size() / cols || col >= cols)
  {
    return;
  }
  bits[row * cols + col] = 1;
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

int toRosRngSeed(uint64_t seed)
{
  constexpr uint64_t kMaxSeed = static_cast<uint64_t>(std::numeric_limits<int>::max());
  const int out = static_cast<int>(seed % kMaxSeed);
  return out == 0 ? 1 : out;
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

Eigen::Quaterniond sampleUniformQuaternion(std::mt19937_64& rng)
{
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const double u1 = unit(rng);
  const double u2 = unit(rng);
  const double u3 = unit(rng);
  const double sqrt1 = std::sqrt(1.0 - u1);
  const double sqrt2 = std::sqrt(u1);
  const double theta1 = 2.0 * M_PI * u2;
  const double theta2 = 2.0 * M_PI * u3;
  Eigen::Quaterniond q(sqrt2 * std::cos(theta2), sqrt1 * std::sin(theta1), sqrt1 * std::cos(theta1),
                       sqrt2 * std::sin(theta2));
  q.normalize();
  return canonicalizeQuat(q);
}

std::vector<Eigen::Quaterniond> selectFarthestQuats(const std::vector<Eigen::Quaterniond>& quats, int max_count);

std::vector<Eigen::Quaterniond> generateHopfQuats(size_t count, uint64_t seed)
{
  const size_t pool_target = std::max<size_t>(count * 8, 1024);
  size_t n_theta = 4;
  size_t n_phi = 8;
  size_t n_psi = 8;
  while (n_theta * n_phi * n_psi < pool_target)
  {
    n_theta += 2;
    n_phi += 4;
    n_psi += 4;
  }

  std::vector<Eigen::Quaterniond> pool;
  pool.reserve(n_theta * n_phi * n_psi);
  const double seed_shift = static_cast<double>(seed % 9973) / 9973.0;
  for (size_t i = 0; i < n_theta; ++i)
  {
    const double u = (static_cast<double>(i) + 0.5) / static_cast<double>(n_theta);
    const double theta = std::asin(std::sqrt(std::min(1.0, std::max(0.0, u))));
    const double ct = std::cos(theta);
    const double st = std::sin(theta);
    for (size_t j = 0; j < n_phi; ++j)
    {
      const double phi = 2.0 * M_PI * std::fmod((static_cast<double>(j) + 0.5) / static_cast<double>(n_phi) +
                                                    0.5 * seed_shift,
                                                1.0);
      for (size_t k = 0; k < n_psi; ++k)
      {
        const double psi = 2.0 * M_PI * std::fmod((static_cast<double>(k) + 0.5) / static_cast<double>(n_psi) +
                                                    0.25 * seed_shift,
                                                1.0);
        Eigen::Quaterniond q(ct * std::cos(phi), ct * std::sin(phi), st * std::cos(psi), st * std::sin(psi));
        q.normalize();
        pool.push_back(canonicalizeQuat(q));
      }
    }
  }
  return selectFarthestQuats(pool, static_cast<int>(count));
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

std::array<Eigen::Quaterniond, 7> makeStencil(const Eigen::Quaterniond& center, double rho)
{
  std::array<Eigen::Quaterniond, 7> out{};
  out[0] = center.normalized();
  const std::array<Eigen::Vector3d, 6> offsets = {
    Eigen::Vector3d(rho, 0.0, 0.0),  Eigen::Vector3d(-rho, 0.0, 0.0),
    Eigen::Vector3d(0.0, rho, 0.0),  Eigen::Vector3d(0.0, -rho, 0.0),
    Eigen::Vector3d(0.0, 0.0, rho),  Eigen::Vector3d(0.0, 0.0, -rho),
  };
  for (size_t i = 0; i < offsets.size(); ++i)
  {
    Eigen::Quaterniond q = expMapSO3(offsets[i]) * center;
    q.normalize();
    out[i + 1] = q;
  }
  return out;
}

Eigen::Quaterniond offsetQuat(const Eigen::Quaterniond& center, const Eigen::Vector3d& offset)
{
  Eigen::Quaterniond q = expMapSO3(offset) * center;
  q.normalize();
  return canonicalizeQuat(q);
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
  reachability_cli::RobotContext robot;
  const moveit::core::JointModelGroup* jmg = nullptr;
  planning_scene::PlanningScenePtr scene;
  collision_detection::CollisionRequest request;
  moveit::core::RobotState state;
  collision_detection::CollisionResult result;
  random_numbers::RandomNumberGenerator rng;
  std::vector<double> joint_positions;

  ThreadContext(reachability_cli::RobotContext robot_context, const moveit::core::JointModelGroup* group, int seed,
                size_t joint_count, const std::string& group_name)
    : robot(std::move(robot_context))
    , jmg(group)
    , scene(robot.scene)
    , state(robot.model)
    , rng(seed)
    , joint_positions(joint_count, 0.0)
  {
    if (!jmg)
    {
      throw std::runtime_error("ThreadContext received null JointModelGroup");
    }
    request.group_name = group_name;
    request.contacts = false;
    request.max_contacts = 0;
    state.setToDefaultValues();
  }
};

bool evaluateIK(const Eigen::Vector3d& pos, const Eigen::Quaterniond& quat, const std::string& ee_link,
                ThreadContext& ctx, const std::vector<Seed>& seeds, double warm_start_mix, int csr_trials,
                int random_trials, double timeout, std::vector<double>& solution)
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
      ctx.state.setJointGroupPositions(ctx.jmg, *seed);
    }
    else
    {
      ctx.state.setToRandomPositions(ctx.jmg, ctx.rng);
    }
    const bool ok = ctx.state.setFromIK(ctx.jmg, pose, ee_link, timeout);
    if (!ok)
    {
      return false;
    }
    ctx.state.update();
    if (!ctx.state.satisfiesBounds(ctx.jmg))
    {
      return false;
    }
    ctx.result.clear();
    ctx.scene->checkSelfCollision(ctx.request, ctx.result, ctx.state);
    if (ctx.result.collision)
    {
      return false;
    }
    ctx.state.copyJointGroupPositions(ctx.jmg, solution);
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
  hid_t anchor_ctx_group = -1;
  hid_t cells_group = -1;
  hid_t cells_csr_group = -1;
  hid_t samples_group = -1;
  hid_t csr_group = -1;
  hid_t meta_group = -1;
  hid_t cell_quat_dset = -1;
  hid_t cell_level_dset = -1;
  hid_t cell_state_dset = -1;
  hid_t cell_phi_graph_dset = -1;
  hid_t cell_anchor_start_dset = -1;
  hid_t quat_dset = -1;
  hid_t phi_dset = -1;
  hid_t label_dset = -1;
  hid_t joint_dset = -1;
  hid_t method_dset = -1;
  hid_t anchor_start_dset = -1;
  bool write_method = true;
  bool write_cells = true;
  size_t cell_count = 0;
  size_t sample_count = 0;
  size_t joint_count = 0;
  size_t chunk_rows = 0;

  bool open(const std::string& path, size_t chunk, size_t joint_dim, bool enable_method, bool enable_cells)
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
    anchor_ctx_group = H5Gcreate2(file, "/anchor_ctx", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    samples_group = H5Gcreate2(file, "/samples", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    csr_group = H5Gcreate2(file, "/csr", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    meta_group = H5Gcreate2(file, "/meta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_cells = enable_cells;
    if (write_cells)
    {
      cells_group = H5Gcreate2(file, "/cells", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      cells_csr_group = H5Gcreate2(file, "/cells_csr", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    }
    if (anchors_group < 0 || anchor_ctx_group < 0 || samples_group < 0 || csr_group < 0 || meta_group < 0 ||
        (write_cells && (cells_group < 0 || cells_csr_group < 0)))
    {
      return false;
    }

    chunk_rows = chunk;
    joint_count = joint_dim;
    write_method = enable_method;

    auto create_cell2d = [&](const char* name, hid_t dtype, hsize_t cols) -> hid_t {
      if (!write_cells)
      {
        return -1;
      }
      hsize_t dims[2] = { 0, cols };
      hsize_t max_dims[2] = { H5S_UNLIMITED, cols };
      hsize_t chunk_dims[2] = { static_cast<hsize_t>(chunk_rows), cols };
      hid_t space = H5Screate_simple(2, dims, max_dims);
      hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
      H5Pset_chunk(plist, 2, chunk_dims);
      hid_t dset = H5Dcreate2(cells_group, name, dtype, space, H5P_DEFAULT, plist, H5P_DEFAULT);
      H5Pclose(plist);
      H5Sclose(space);
      return dset;
    };

    auto create_cell1d = [&](const char* name, hid_t dtype) -> hid_t {
      if (!write_cells)
      {
        return -1;
      }
      hsize_t dims[1] = { 0 };
      hsize_t max_dims[1] = { H5S_UNLIMITED };
      hsize_t chunk_dims[1] = { static_cast<hsize_t>(chunk_rows) };
      hid_t space = H5Screate_simple(1, dims, max_dims);
      hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
      H5Pset_chunk(plist, 1, chunk_dims);
      hid_t dset = H5Dcreate2(cells_group, name, dtype, space, H5P_DEFAULT, plist, H5P_DEFAULT);
      H5Pclose(plist);
      H5Sclose(space);
      return dset;
    };

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
    if (write_cells)
    {
      cell_quat_dset = create_cell2d("quat_center", H5T_IEEE_F32LE, 4);
      cell_level_dset = create_cell1d("level", H5T_STD_U8LE);
      cell_state_dset = create_cell1d("state", H5T_STD_U8LE);
      cell_phi_graph_dset = create_cell1d("phi_graph", H5T_IEEE_F32LE);
    }
    return quat_dset >= 0 && phi_dset >= 0 && label_dset >= 0 && (!write_method || method_dset >= 0) &&
           (!write_cells || (cell_quat_dset >= 0 && cell_level_dset >= 0 && cell_state_dset >= 0 &&
                             cell_phi_graph_dset >= 0));
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

  bool initCellAnchorStart()
  {
    if (!write_cells || cells_csr_group < 0)
    {
      return false;
    }
    hsize_t dims[1] = { 0 };
    hsize_t max_dims[1] = { H5S_UNLIMITED };
    hsize_t chunk_dims[1] = { static_cast<hsize_t>(chunk_rows) };
    hid_t space = H5Screate_simple(1, dims, max_dims);
    hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(plist, 1, chunk_dims);
    cell_anchor_start_dset =
        H5Dcreate2(cells_csr_group, "anchor_start", H5T_STD_U64LE, space, H5P_DEFAULT, plist, H5P_DEFAULT);
    H5Pclose(plist);
    H5Sclose(space);
    return cell_anchor_start_dset >= 0;
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

  bool appendCellAnchorStart(uint64_t value)
  {
    if (cell_anchor_start_dset < 0)
    {
      return false;
    }
    hid_t space = H5Dget_space(cell_anchor_start_dset);
    hsize_t dims[1] = { 0 };
    H5Sget_simple_extent_dims(space, dims, nullptr);
    H5Sclose(space);

    hsize_t new_dims[1] = { dims[0] + 1 };
    if (H5Dset_extent(cell_anchor_start_dset, new_dims) < 0)
    {
      return false;
    }
    hid_t filespace = H5Dget_space(cell_anchor_start_dset);
    hsize_t start[1] = { dims[0] };
    hsize_t block[1] = { 1 };
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, block, nullptr);
    hid_t memspace = H5Screate_simple(1, block, nullptr);
    const bool ok = H5Dwrite(cell_anchor_start_dset, H5T_STD_U64LE, memspace, filespace, H5P_DEFAULT, &value) >= 0;
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

  bool appendCells(const std::vector<CellOutput>& cells)
  {
    if (!write_cells || cells.empty())
    {
      return true;
    }
    const size_t count = cells.size();
    std::vector<float> quat_buf(count * 4, 0.0f);
    std::vector<uint8_t> level_buf(count, 0);
    std::vector<uint8_t> state_buf(count, 3);
    std::vector<float> phi_buf(count, std::numeric_limits<float>::quiet_NaN());
    for (size_t i = 0; i < count; ++i)
    {
      quat_buf[i * 4 + 0] = static_cast<float>(cells[i].quat.x());
      quat_buf[i * 4 + 1] = static_cast<float>(cells[i].quat.y());
      quat_buf[i * 4 + 2] = static_cast<float>(cells[i].quat.z());
      quat_buf[i * 4 + 3] = static_cast<float>(cells[i].quat.w());
      level_buf[i] = cells[i].level;
      state_buf[i] = cells[i].state;
      phi_buf[i] = cells[i].phi_graph;
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

    if (!append2d(cell_quat_dset, H5T_IEEE_F32LE, quat_buf.data(), 4))
    {
      return false;
    }
    if (!append1d(cell_level_dset, H5T_STD_U8LE, level_buf.data()))
    {
      return false;
    }
    if (!append1d(cell_state_dset, H5T_STD_U8LE, state_buf.data()))
    {
      return false;
    }
    if (!append1d(cell_phi_graph_dset, H5T_IEEE_F32LE, phi_buf.data()))
    {
      return false;
    }

    cell_count += count;
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

  template <typename T>
  bool writeScalarValue(hid_t group, const std::string& name, hid_t dtype, T value)
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
    if (cell_quat_dset >= 0)
    {
      H5Dclose(cell_quat_dset);
    }
    if (cell_level_dset >= 0)
    {
      H5Dclose(cell_level_dset);
    }
    if (cell_state_dset >= 0)
    {
      H5Dclose(cell_state_dset);
    }
    if (cell_phi_graph_dset >= 0)
    {
      H5Dclose(cell_phi_graph_dset);
    }
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
    if (anchor_ctx_group >= 0)
    {
      H5Gclose(anchor_ctx_group);
    }
    if (cells_group >= 0)
    {
      H5Gclose(cells_group);
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
    if (cell_anchor_start_dset >= 0)
    {
      H5Dclose(cell_anchor_start_dset);
      cell_anchor_start_dset = -1;
    }
    if (csr_group >= 0)
    {
      H5Gclose(csr_group);
    }
    if (cells_csr_group >= 0)
    {
      H5Gclose(cells_csr_group);
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
  std::vector<std::array<float, 4>> coverage_bin_quats;
  try
  {
    std::vector<hsize_t> bin_dims;
    const std::vector<float> bin_data = readDataset<float>(base_file, "/grid/orientation_bins", &bin_dims);
    if (bin_dims.size() == 2 && bin_dims[1] == 4 && bin_data.size() == static_cast<size_t>(bin_dims[0] * 4))
    {
      coverage_bin_quats.reserve(static_cast<size_t>(bin_dims[0]));
      for (size_t i = 0; i < static_cast<size_t>(bin_dims[0]); ++i)
      {
        coverage_bin_quats.push_back({ bin_data[i * 4 + 0], bin_data[i * 4 + 1], bin_data[i * 4 + 2],
                                       bin_data[i * 4 + 3] });
      }
    }
  }
  catch (const std::exception&)
  {
    RCLCPP_WARN(logger, "Base HDF5 has no /grid/orientation_bins; anchor_ctx coverage_bits may be unavailable.");
  }

  std::vector<hsize_t> orientation_bits_dims;
  hid_t orientation_bits_dset = -1;
  size_t orientation_word_count = 0;
  bool has_orientation_bits = false;
  if (datasetExists(base_file, "/grid/orientation_bits"))
  {
    orientation_bits_dset = H5Dopen2(base_file, "/grid/orientation_bits", H5P_DEFAULT);
    if (orientation_bits_dset >= 0)
    {
      hid_t bits_space = H5Dget_space(orientation_bits_dset);
      const int rank = H5Sget_simple_extent_ndims(bits_space);
      orientation_bits_dims.assign(static_cast<size_t>(rank), 0);
      H5Sget_simple_extent_dims(bits_space, orientation_bits_dims.data(), nullptr);
      H5Sclose(bits_space);

      if (orientation_bits_dims.size() == 4 && orientation_bits_dims[0] == grid.nx &&
          orientation_bits_dims[1] == grid.ny && orientation_bits_dims[2] == grid.nz &&
          orientation_bits_dims[3] > 0)
      {
        orientation_word_count = static_cast<size_t>(orientation_bits_dims[3]);
      }
      else if (orientation_bits_dims.size() == 2 && orientation_bits_dims[0] == grid_size &&
               orientation_bits_dims[1] > 0)
      {
        orientation_word_count = static_cast<size_t>(orientation_bits_dims[1]);
      }
      else if (orientation_bits_dims.size() == 1 && orientation_bits_dims[0] == grid_size)
      {
        orientation_word_count = 1;
      }
      else
      {
        RCLCPP_WARN(logger, "/grid/orientation_bits shape does not match grid; falling back to reservoir-derived "
                            "anchor_ctx coverage_bits.");
        H5Dclose(orientation_bits_dset);
        orientation_bits_dset = -1;
      }
      has_orientation_bits = orientation_bits_dset >= 0 && orientation_word_count > 0;
    }
  }
  std::vector<uint32_t> counts = readDataset<uint32_t>(base_file, "/grid/voxel_counts");
  bool has_seed_csr = true;
  std::vector<uint64_t> voxel_start;
  try
  {
    voxel_start = readDataset<uint64_t>(base_file, "/csr/voxel_start");
  }
  catch (const std::exception&)
  {
    has_seed_csr = false;
    RCLCPP_WARN(logger, "Base HDF5 has no /csr/voxel_start; orientation mining will use random FK seeds only.");
  }
  std::string index_dtype = readStringDataset(base_file, "/csr/index_dtype", "uint32");
  bool sample_index_is_u64 = false;
  std::vector<uint64_t> sample_index_u64;
  std::vector<uint32_t> sample_index_u32;
  if (has_seed_csr && index_dtype == "uint64")
  {
    sample_index_is_u64 = true;
    sample_index_u64 = readDataset<uint64_t>(base_file, "/csr/sample_index");
  }
  else if (has_seed_csr)
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
  random_numbers::RandomNumberGenerator quant_rng(toRosRngSeed(cfg.anchor_quantile_seed));
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
    random_numbers::RandomNumberGenerator rng(
        toRosRngSeed(cfg.anchor_quota_seed_base + static_cast<uint64_t>(tid) * cfg.anchor_quota_seed_stride));

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
      random_numbers::RandomNumberGenerator rng(
          toRosRngSeed(cfg.anchor_global_seed_base + static_cast<uint64_t>(tid) * cfg.anchor_global_seed_stride));

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
  std::vector<uint8_t> anchor_selected(total_anchors, 0);
  std::vector<uint8_t> anchor_full_reachable(total_anchors, 0);
  size_t start_anchor = 0;
  size_t process_count = 0;
  if (!cfg.debug_anchor_indices.empty())
  {
    for (const int raw_idx : cfg.debug_anchor_indices)
    {
      if (raw_idx < 0)
      {
        continue;
      }
      const size_t idx = static_cast<size_t>(raw_idx);
      if (idx >= total_anchors || anchor_selected[idx] != 0)
      {
        continue;
      }
      anchor_selected[idx] = 1;
      ++process_count;
    }
    RCLCPP_INFO(logger, "Anchors selected: %zu (processing %zu explicit indices)", total_anchors, process_count);
  }
  else
  {
    if (cfg.debug_start_anchor > 0)
    {
      start_anchor = std::min(static_cast<size_t>(cfg.debug_start_anchor), total_anchors);
    }
    process_count = total_anchors - start_anchor;
    if (cfg.debug_max_anchors > 0)
    {
      process_count = std::min(process_count, static_cast<size_t>(cfg.debug_max_anchors));
    }
    for (size_t i = start_anchor; i < start_anchor + process_count; ++i)
    {
      anchor_selected[i] = 1;
    }
    RCLCPP_INFO(logger, "Anchors selected: %zu (processing %zu, start=%zu)", total_anchors, process_count,
                start_anchor);
  }

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

  auto configure_solver = [&](const moveit::core::RobotModelPtr& model,
                              const moveit::core::JointModelGroup* group) -> bool {
    if (!group)
    {
      return false;
    }
    auto solver_allocator = [node, model, cfg](const moveit::core::JointModelGroup* solver_group)
        -> kinematics::KinematicsBasePtr {
      auto solver = std::make_shared<kdl_kinematics_plugin::KDLKinematicsPlugin>();
      const std::vector<std::string> tips{ cfg.ee_link };
      if (!solver->initialize(node, *model, solver_group->getName(), cfg.base_link, tips, cfg.search_discretization))
      {
        RCLCPP_ERROR(node->get_logger(), "Failed to initialize KDL solver for group '%s'",
                     solver_group->getName().c_str());
        return kinematics::KinematicsBasePtr();
      }
      return kinematics::KinematicsBasePtr(solver);
    };
    const_cast<moveit::core::JointModelGroup*>(group)->setSolverAllocators(solver_allocator);
    const_cast<moveit::core::JointModelGroup*>(group)->setDefaultIKTimeout(cfg.ik_timeout_coarse);
    return true;
  };

  configure_solver(robot_model, jmg);

  std::vector<ThreadContext> thread_contexts;
  thread_contexts.reserve(static_cast<size_t>(thread_count));
  for (int i = 0; i < thread_count; ++i)
  {
    reachability_cli::RobotContext thread_robot;
    try
    {
      thread_robot = reachability_cli::loadRobotFromFiles(cfg.urdf_path, cfg.srdf_path);
    }
    catch (const std::exception& ex)
    {
      RCLCPP_ERROR(logger, "Failed to load per-thread robot model %d: %s", i, ex.what());
      H5Fclose(base_file);
      rclcpp::shutdown();
      return 1;
    }
    const auto* thread_jmg = thread_robot.model->getJointModelGroup(cfg.group_name);
    if (!configure_solver(thread_robot.model, thread_jmg))
    {
      RCLCPP_ERROR(logger, "JointModelGroup '%s' not found in per-thread robot model %d", cfg.group_name.c_str(), i);
      H5Fclose(base_file);
      rclcpp::shutdown();
      return 1;
    }
    const int thread_seed =
        toRosRngSeed(cfg.thread_seed_base + static_cast<uint64_t>(i) * cfg.thread_seed_stride);
    thread_contexts.emplace_back(std::move(thread_robot), thread_jmg, thread_seed, joint_count, cfg.group_name);
  }

  const int so3_base_cells = std::max(1, cfg.so3_base_cells);
  const int so3_max_depth = std::max(0, cfg.so3_max_depth);
  const double base_radius =
      std::max(0.32, cfg.so3_leaf_radius * std::pow(2.0, static_cast<double>(so3_max_depth)));
  if (cfg.so3_grid != "hopf")
  {
    RCLCPP_WARN(logger, "Unknown so3_grid='%s'; using hopf.", cfg.so3_grid.c_str());
    cfg.so3_grid = "hopf";
  }
  if (cfg.so3_split_factor != 8)
  {
    RCLCPP_WARN(logger, "so3_split_factor=%d requested; current implementation uses fixed 8-way tangent split.",
                cfg.so3_split_factor);
    cfg.so3_split_factor = 8;
  }
  RCLCPP_INFO(logger, "SO3 grid: %s base_cells=%d depth=%d base_radius=%.4f leaf_radius=%.4f",
              cfg.so3_grid.c_str(), so3_base_cells, so3_max_depth, base_radius, cfg.so3_leaf_radius);
  const std::vector<Eigen::Quaterniond> base_quats = generateHopfQuats(static_cast<size_t>(so3_base_cells), 0);

  std::vector<Eigen::Quaterniond> q_ref;
  if (cfg.q_ref_size > 0)
  {
    const int q_pool = std::max(cfg.q_ref_pool, cfg.q_ref_size);
    const uint64_t q_seed = cfg.q_ref_seed;
    RCLCPP_INFO(logger, "Q_ref: pool=%d size=%d (seed=%llu)", q_pool, cfg.q_ref_size,
                static_cast<unsigned long long>(q_seed));
    const auto q_pool_quats = generateHopfQuats(static_cast<size_t>(q_pool), q_seed);
    q_ref = selectFarthestQuats(q_pool_quats, cfg.q_ref_size);
  }
  uint64_t full_validation_seed_base = cfg.full_reachable_validation_seed;
  if (full_validation_seed_base == 0)
  {
    full_validation_seed_base = 20260429ULL;
    RCLCPP_WARN(logger, "sampling.full_reachable_validation_seed=0 is not reproducible in the config; using "
                        "deterministic fallback seed %llu.",
                static_cast<unsigned long long>(full_validation_seed_base));
  }

  Hdf5Writer writer;
  if (!writer.open(cfg.output_path, cfg.hdf5_chunk, cfg.write_joint ? joint_count : 0, cfg.write_method,
                   cfg.write_cells))
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
  if (cfg.write_cells)
  {
    if (!writer.initCellAnchorStart())
    {
      RCLCPP_ERROR(logger, "Failed to init streaming /cells_csr/anchor_start.");
      writer.close();
      H5Fclose(base_file);
      rclcpp::shutdown();
      return 1;
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
  writer.writeString(writer.meta_group, "so3_grid", cfg.so3_grid);
  writer.writeScalarValue<int32_t>(writer.meta_group, "so3_base_cells", H5T_STD_I32LE, so3_base_cells);
  writer.writeScalarValue<int32_t>(writer.meta_group, "so3_split_factor", H5T_STD_I32LE, cfg.so3_split_factor);
  writer.writeScalarValue<int32_t>(writer.meta_group, "so3_max_depth", H5T_STD_I32LE, so3_max_depth);
  writer.writeScalar(writer.meta_group, "so3_base_radius", H5T_IEEE_F64LE, base_radius);
  writer.writeScalar(writer.meta_group, "so3_leaf_radius", H5T_IEEE_F64LE, cfg.so3_leaf_radius);
  writer.writeScalar(writer.meta_group, "so3_stencil_radius_ratio", H5T_IEEE_F64LE, cfg.so3_stencil_radius_ratio);
  writer.writeScalar(writer.meta_group, "so3_refine_band", H5T_IEEE_F64LE, cfg.so3_refine_band);
  writer.writeScalar(writer.meta_group, "delta_boundary", H5T_IEEE_F64LE, cfg.delta_boundary);
  writer.writeScalarValue<int32_t>(writer.meta_group, "boundary_refine_steps", H5T_STD_I32LE,
                                   cfg.boundary_refine_steps);
  writer.writeScalarValue<int32_t>(writer.meta_group, "max_boundary_brackets", H5T_STD_I32LE,
                                   cfg.max_boundary_brackets);
  writer.writeScalarValue<int32_t>(writer.meta_group, "full_reachable_validation_samples", H5T_STD_I32LE,
                                   cfg.full_reachable_validation_samples);
  writer.writeScalarValue<uint64_t>(writer.meta_group, "full_reachable_validation_seed", H5T_STD_U64LE,
                                    cfg.full_reachable_validation_seed);
  writer.writeScalarValue<uint64_t>(writer.meta_group, "full_reachable_validation_effective_seed_base",
                                    H5T_STD_U64LE, full_validation_seed_base);
  writer.writeScalarValue<uint64_t>(writer.meta_group, "q_ref_seed", H5T_STD_U64LE, cfg.q_ref_seed);
  writer.writeScalarValue<uint64_t>(writer.meta_group, "so3_base_seed", H5T_STD_U64LE, 0ULL);
  writer.writeScalarValue<uint64_t>(writer.meta_group, "anchor_quantile_seed", H5T_STD_U64LE,
                                    cfg.anchor_quantile_seed);
  writer.writeScalarValue<uint64_t>(writer.meta_group, "anchor_quota_seed_base", H5T_STD_U64LE,
                                    cfg.anchor_quota_seed_base);
  writer.writeScalarValue<uint64_t>(writer.meta_group, "anchor_quota_seed_stride", H5T_STD_U64LE,
                                    cfg.anchor_quota_seed_stride);
  writer.writeScalarValue<uint64_t>(writer.meta_group, "anchor_global_seed_base", H5T_STD_U64LE,
                                    cfg.anchor_global_seed_base);
  writer.writeScalarValue<uint64_t>(writer.meta_group, "anchor_global_seed_stride", H5T_STD_U64LE,
                                    cfg.anchor_global_seed_stride);
  writer.writeScalarValue<uint64_t>(writer.meta_group, "thread_seed_base", H5T_STD_U64LE, cfg.thread_seed_base);
  writer.writeScalarValue<uint64_t>(writer.meta_group, "thread_seed_stride", H5T_STD_U64LE, cfg.thread_seed_stride);
  writer.writeScalarValue<uint64_t>(writer.meta_group, "fallback_seed_base", H5T_STD_U64LE,
                                    cfg.fallback_seed_base);
  writer.writeScalarValue<uint64_t>(writer.meta_group, "fallback_seed_stride", H5T_STD_U64LE,
                                    cfg.fallback_seed_stride);
  writer.writeScalarValue<int32_t>(writer.meta_group, "ik_csr_trials_coarse", H5T_STD_I32LE,
                                   cfg.ik_csr_trials_coarse);
  writer.writeScalarValue<int32_t>(writer.meta_group, "ik_random_trials_coarse", H5T_STD_I32LE,
                                   cfg.ik_random_trials_coarse);
  writer.writeScalar(writer.meta_group, "ik_timeout_coarse", H5T_IEEE_F64LE, cfg.ik_timeout_coarse);
  writer.writeScalarValue<int32_t>(writer.meta_group, "ik_csr_trials_refine", H5T_STD_I32LE,
                                   cfg.ik_csr_trials_refine);
  writer.writeScalarValue<int32_t>(writer.meta_group, "ik_random_trials_refine", H5T_STD_I32LE,
                                   cfg.ik_random_trials_refine);
  writer.writeScalar(writer.meta_group, "ik_timeout_refine", H5T_IEEE_F64LE, cfg.ik_timeout_refine);
  writer.writeScalarValue<int32_t>(writer.meta_group, "ik_csr_trials_bisect", H5T_STD_I32LE,
                                   cfg.ik_csr_trials_bisect);
  writer.writeScalarValue<int32_t>(writer.meta_group, "ik_random_trials_bisect", H5T_STD_I32LE,
                                   cfg.ik_random_trials_bisect);
  writer.writeScalar(writer.meta_group, "ik_timeout_bisect", H5T_IEEE_F64LE, cfg.ik_timeout_bisect);
  if (!cfg.shell_deltas.empty())
  {
    std::vector<double> shell = cfg.shell_deltas;
    writer.writeArray(writer.meta_group, "shell_deltas", H5T_IEEE_F64LE, { shell.size() }, shell.data());
  }
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
    writer.writeScalarValue<int32_t>(writer.meta_group, "q_ref_size", H5T_STD_I32LE, cfg.q_ref_size);
    writer.writeScalarValue<int32_t>(writer.meta_group, "q_ref_pool", H5T_STD_I32LE, cfg.q_ref_pool);
  }
  writer.writeScalar(writer.meta_group, "phi_full", H5T_IEEE_F64LE, cfg.phi_full);

  auto q_ref_bucket = [&](const Eigen::Quaterniond& q) -> size_t {
    if (q_ref.empty())
    {
      return 0;
    }
    size_t best = 0;
    double best_dot = -1.0;
    for (size_t i = 0; i < q_ref.size(); ++i)
    {
      const double dot = std::abs(q.dot(q_ref[i]));
      if (dot > best_dot)
      {
        best_dot = dot;
        best = i;
      }
    }
    return best;
  };

  std::vector<uint64_t> anchor_ids;
  std::vector<float> anchor_pos;
  std::vector<float> anchor_s;
  std::vector<float> anchor_c;
  std::vector<float> anchor_g;
  std::vector<uint32_t> anchor_n;
  constexpr size_t kAnchorCtxScalarDim = 8;
  size_t coverage_bit_count = coverage_bin_quats.empty() ? size_t{ 64 } : coverage_bin_quats.size();
  if (has_orientation_bits)
  {
    coverage_bit_count = std::min(coverage_bit_count, orientation_word_count * size_t{ 64 });
  }
  if (coverage_bit_count == 0)
  {
    coverage_bit_count = 64;
  }
  std::vector<float> anchor_ctx_scalar(anchors.size() * kAnchorCtxScalarDim, 0.0f);
  std::vector<uint8_t> anchor_ctx_coverage_bits(anchors.size() * coverage_bit_count, 0);

  auto xyz_from_index = [&](size_t idx, size_t& x, size_t& y, size_t& z) {
    x = idx / (grid.ny * grid.nz);
    const size_t rem = idx % (grid.ny * grid.nz);
    y = rem / grid.nz;
    z = rem % grid.nz;
  };

  auto sbar_at = [&](size_t x, size_t y, size_t z) -> double {
    const size_t idx = grid.index(x, y, z);
    return static_cast<double>(sdf[idx]) / s_ref;
  };

  auto finite_diff_sbar = [&](size_t x, size_t y, size_t z, int axis) -> double {
    size_t xm = x;
    size_t xp = x;
    size_t ym = y;
    size_t yp = y;
    size_t zm = z;
    size_t zp = z;
    if (axis == 0)
    {
      xm = x > 0 ? x - 1 : x;
      xp = x + 1 < grid.nx ? x + 1 : x;
    }
    else if (axis == 1)
    {
      ym = y > 0 ? y - 1 : y;
      yp = y + 1 < grid.ny ? y + 1 : y;
    }
    else
    {
      zm = z > 0 ? z - 1 : z;
      zp = z + 1 < grid.nz ? z + 1 : z;
    }
    const int step = axis == 0 ? static_cast<int>(xp) - static_cast<int>(xm) :
                     axis == 1 ? static_cast<int>(yp) - static_cast<int>(ym) :
                                 static_cast<int>(zp) - static_cast<int>(zm);
    if (step == 0)
    {
      return 0.0;
    }
    return (sbar_at(xp, yp, zp) - sbar_at(xm, ym, zm)) /
           (static_cast<double>(step) * grid.voxel_size);
  };

  auto kappa_proxy = [&](size_t x, size_t y, size_t z) -> double {
    const double center = sbar_at(x, y, z);
    double accum = 0.0;
    auto add_neighbor = [&](bool valid, size_t nx, size_t ny, size_t nz) {
      if (valid)
      {
        accum += sbar_at(nx, ny, nz) - center;
      }
    };
    add_neighbor(x > 0, x > 0 ? x - 1 : x, y, z);
    add_neighbor(x + 1 < grid.nx, x + 1 < grid.nx ? x + 1 : x, y, z);
    add_neighbor(y > 0, x, y > 0 ? y - 1 : y, z);
    add_neighbor(y + 1 < grid.ny, x, y + 1 < grid.ny ? y + 1 : y, z);
    add_neighbor(z > 0, x, y, z > 0 ? z - 1 : z);
    add_neighbor(z + 1 < grid.nz, x, y, z + 1 < grid.nz ? z + 1 : z);
    return accum / std::max(1e-18, grid.voxel_size * grid.voxel_size);
  };

  for (size_t ai = 0; ai < anchors.size(); ++ai)
  {
    const auto& a = anchors[ai];
    anchor_ids.push_back(static_cast<uint64_t>(a.idx));
    anchor_pos.push_back(a.pos.x());
    anchor_pos.push_back(a.pos.y());
    anchor_pos.push_back(a.pos.z());
    anchor_s.push_back(a.s_v);
    anchor_c.push_back(a.c_v);
    anchor_g.push_back(a.g_v);
    anchor_n.push_back(a.n_seed);

    size_t x = 0;
    size_t y = 0;
    size_t z = 0;
    xyz_from_index(a.idx, x, y, z);
    const double gx = finite_diff_sbar(x, y, z, 0);
    const double gy = finite_diff_sbar(x, y, z, 1);
    const double gz = finite_diff_sbar(x, y, z, 2);
    const double grad_norm = std::sqrt(gx * gx + gy * gy + gz * gz);
    const double kappa = kappa_proxy(x, y, z);
    const size_t scalar_offset = ai * kAnchorCtxScalarDim;
    anchor_ctx_scalar[scalar_offset + 0] = static_cast<float>(gx);
    anchor_ctx_scalar[scalar_offset + 1] = static_cast<float>(gy);
    anchor_ctx_scalar[scalar_offset + 2] = static_cast<float>(gz);
    anchor_ctx_scalar[scalar_offset + 3] = static_cast<float>(grad_norm);
    anchor_ctx_scalar[scalar_offset + 4] = static_cast<float>(kappa);
    anchor_ctx_scalar[scalar_offset + 5] = static_cast<float>(a.n_seed);
    anchor_ctx_scalar[scalar_offset + 6] = a.c_v;
    anchor_ctx_scalar[scalar_offset + 7] = a.g_v;
  }

  std::string coverage_bits_source = "unavailable_zero_filled";
  bool coverage_bits_exact = false;
  if (has_orientation_bits)
  {
    coverage_bits_source = "/grid/orientation_bits";
    coverage_bits_exact = true;
    for (size_t ai = 0; ai < anchors.size(); ++ai)
    {
      size_t ax = 0;
      size_t ay = 0;
      size_t az = 0;
      xyz_from_index(anchors[ai].idx, ax, ay, az);
      std::vector<uint64_t> words(orientation_word_count, 0);
      hid_t filespace = H5Dget_space(orientation_bits_dset);
      hid_t memspace = -1;
      if (orientation_bits_dims.size() == 4)
      {
        hsize_t start[4] = { static_cast<hsize_t>(ax), static_cast<hsize_t>(ay), static_cast<hsize_t>(az), 0 };
        hsize_t count[4] = { 1, 1, 1, static_cast<hsize_t>(orientation_word_count) };
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);
        hsize_t mem_dims[1] = { static_cast<hsize_t>(orientation_word_count) };
        memspace = H5Screate_simple(1, mem_dims, nullptr);
      }
      else if (orientation_bits_dims.size() == 2)
      {
        hsize_t start[2] = { static_cast<hsize_t>(anchors[ai].idx), 0 };
        hsize_t count[2] = { 1, static_cast<hsize_t>(orientation_word_count) };
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);
        hsize_t mem_dims[1] = { static_cast<hsize_t>(orientation_word_count) };
        memspace = H5Screate_simple(1, mem_dims, nullptr);
      }
      else
      {
        hsize_t start[1] = { static_cast<hsize_t>(anchors[ai].idx) };
        hsize_t count[1] = { 1 };
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);
        hsize_t mem_dims[1] = { 1 };
        memspace = H5Screate_simple(1, mem_dims, nullptr);
      }
      const bool read_ok =
          memspace >= 0 && H5Dread(orientation_bits_dset, H5T_STD_U64LE, memspace, filespace, H5P_DEFAULT,
                                   words.data()) >= 0;
      if (memspace >= 0)
      {
        H5Sclose(memspace);
      }
      H5Sclose(filespace);
      if (!read_ok)
      {
        continue;
      }
      for (size_t b = 0; b < coverage_bit_count; ++b)
      {
        const size_t w = b / 64;
        const size_t bit = b % 64;
        if (w < words.size() && ((words[w] >> bit) & 1ULL) != 0ULL)
        {
          setBit(anchor_ctx_coverage_bits, ai, coverage_bit_count, b);
        }
      }
    }
    if (orientation_bits_dset >= 0)
    {
      H5Dclose(orientation_bits_dset);
      orientation_bits_dset = -1;
    }
  }
  else if (has_seed_csr && !coverage_bin_quats.empty())
  {
    hid_t quat_dset = H5Dopen2(base_file, "/samples/quat", H5P_DEFAULT);
    if (quat_dset >= 0)
    {
      coverage_bits_source = "reservoir_samples_nearest_/grid/orientation_bins";
      for (size_t ai = 0; ai < anchors.size(); ++ai)
      {
        const uint64_t voxel_idx = static_cast<uint64_t>(anchors[ai].idx);
        if (voxel_idx + 1 >= voxel_start.size())
        {
          continue;
        }
        const uint64_t start = voxel_start[static_cast<size_t>(voxel_idx)];
        const uint64_t end = voxel_start[static_cast<size_t>(voxel_idx + 1)];
        if (end <= start || start >= sample_index_size)
        {
          continue;
        }
        const size_t end_cap = static_cast<size_t>(std::min<uint64_t>(end, sample_index_size));
        for (size_t si = static_cast<size_t>(start); si < end_cap; ++si)
        {
          const uint64_t row = sample_index_is_u64 ? sample_index_u64[si] : sample_index_u32[si];
          hsize_t start_row[2] = { static_cast<hsize_t>(row), 0 };
          hsize_t count_row[2] = { 1, 4 };
          std::array<float, 4> qbuf{};
          hid_t qspace = H5Dget_space(quat_dset);
          H5Sselect_hyperslab(qspace, H5S_SELECT_SET, start_row, nullptr, count_row, nullptr);
          hid_t qmem = H5Screate_simple(2, count_row, nullptr);
          const bool ok = H5Dread(quat_dset, H5T_IEEE_F32LE, qmem, qspace, H5P_DEFAULT, qbuf.data()) >= 0;
          H5Sclose(qmem);
          H5Sclose(qspace);
          if (!ok)
          {
            continue;
          }
          Eigen::Quaterniond q(qbuf[3], qbuf[0], qbuf[1], qbuf[2]);
          q.normalize();
          const int bin = nearestCoverageBin(q, coverage_bin_quats);
          if (bin >= 0)
          {
            setBit(anchor_ctx_coverage_bits, ai, coverage_bit_count, static_cast<size_t>(bin));
          }
        }
      }
      H5Dclose(quat_dset);
      RCLCPP_WARN(logger,
                  "Base HDF5 has no /grid/orientation_bits; /anchor_ctx/coverage_bits was reconstructed from "
                  "stored reservoir samples and is not the full Stage-1 accepted-hit bitset.");
    }
  }
  else
  {
    RCLCPP_WARN(logger, "No source available for /anchor_ctx/coverage_bits; writing zeros.");
  }

  writer.writeArray(writer.anchors_group, "voxel_id", H5T_STD_U64LE, { anchor_ids.size() }, anchor_ids.data());
  writer.writeArray(writer.anchors_group, "pos", H5T_IEEE_F32LE, { anchor_ids.size(), 3 }, anchor_pos.data());
  writer.writeArray(writer.anchors_group, "s_v", H5T_IEEE_F32LE, { anchor_ids.size() }, anchor_s.data());
  writer.writeArray(writer.anchors_group, "c_v", H5T_IEEE_F32LE, { anchor_ids.size() }, anchor_c.data());
  writer.writeArray(writer.anchors_group, "g_v", H5T_IEEE_F32LE, { anchor_ids.size() }, anchor_g.data());
  writer.writeArray(writer.anchors_group, "n_seed", H5T_STD_U32LE, { anchor_ids.size() }, anchor_n.data());
  writer.writeArray(writer.anchors_group, "selected", H5T_STD_U8LE, { anchor_selected.size() }, anchor_selected.data());
  writer.writeArray(writer.anchor_ctx_group, "scalar", H5T_IEEE_F32LE,
                    { anchor_ids.size(), static_cast<hsize_t>(kAnchorCtxScalarDim) }, anchor_ctx_scalar.data());
  writer.writeArray(writer.anchor_ctx_group, "coverage_bits", H5T_STD_U8LE,
                    { anchor_ids.size(), static_cast<hsize_t>(coverage_bit_count) },
                    anchor_ctx_coverage_bits.data());
  writer.writeString(writer.meta_group, "anchor_context_spec_json",
                     "{\"scalar_fields\":[\"grad_s_x\",\"grad_s_y\",\"grad_s_z\",\"grad_norm\","
                     "\"kappa_proxy\",\"n_v\",\"c_v\",\"g_v\"],\"s_bar\":\"sdf/s_ref_p90\","
                     "\"gradient\":\"finite_difference_over_physical_voxel_size\","
                     "\"kappa_proxy\":\"six_neighbor_laplacian_over_physical_voxel_size_squared\","
                     "\"coverage_bits\":\"Stage-1 orientation coverage bitset aligned to anchors\"}");
  writer.writeString(writer.meta_group, "anchor_context_coverage_bits_source", coverage_bits_source);
  writer.writeScalarValue<int8_t>(writer.meta_group, "anchor_context_coverage_bits_exact", H5T_STD_I8LE,
                                  coverage_bits_exact ? 1 : 0);
  writer.writeScalarValue<uint64_t>(writer.meta_group, "anchor_context_coverage_bits_count", H5T_STD_U64LE,
                                    static_cast<uint64_t>(coverage_bit_count));

  if (cfg.anchor_only)
  {
    writer.writeArray(writer.anchors_group, "full_reachable", H5T_STD_U8LE,
                      { anchor_full_reachable.size() }, anchor_full_reachable.data());
    writer.writeScalarValue<int8_t>(writer.meta_group, "anchor_only", H5T_STD_I8LE, 1);
    RCLCPP_INFO(logger, "Anchor-only mode enabled. Wrote anchors to %s", cfg.output_path.c_str());
    writer.close();
    H5Fclose(base_file);
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(0);
  }

  std::vector<uint64_t> anchor_start;
  std::vector<uint64_t> cell_anchor_start;
  anchor_start.reserve(anchors.size() + 1);
  cell_anchor_start.reserve(anchors.size() + 1);
  anchor_start.push_back(0);
  cell_anchor_start.push_back(0);
  if (cfg.write_csr && cfg.stream_csr)
  {
    writer.appendAnchorStart(0);
    if (cfg.flush_per_anchor)
    {
      writer.flush();
    }
  }
  if (cfg.write_cells)
  {
    writer.appendCellAnchorStart(0);
  }

  auto readSeeds = [&](uint64_t voxel_idx) -> std::vector<Seed> {
    std::vector<Seed> seeds;
    if (!has_seed_csr)
    {
      return seeds;
    }
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

  auto cell_sign = [](So3CellState state) -> int {
    if (state == So3CellState::kInside)
    {
      return 1;
    }
    if (state == So3CellState::kOutside)
    {
      return -1;
    }
    return 0;
  };

  auto certify_cell = [&](So3Cell& cell, const Eigen::Vector3d& pos, const std::vector<Seed>& seeds,
                          const std::vector<So3Cell>& all_cells, ThreadContext& ctx, int csr_trials,
                          int random_trials, double timeout) {
    std::vector<Seed> eval_seeds = seeds;
    if (cell.parent >= 0)
    {
      const auto& parent = all_cells[static_cast<size_t>(cell.parent)];
      if (parent.has_center_joint)
      {
        Seed parent_seed;
        parent_seed.quat = parent.center;
        parent_seed.joint = parent.center_joint;
        eval_seeds.push_back(parent_seed);
      }
    }

    const double rho = std::max(1e-6, cell.radius * cfg.so3_stencil_radius_ratio);
    cell.stencil_quat = makeStencil(cell.center, rho);
    int pos_count = 0;
    int neg_count = 0;
    cell.has_center_joint = false;
    cell.center_joint.clear();
    for (size_t si = 0; si < cell.stencil_quat.size(); ++si)
    {
      std::vector<double> solution(joint_count, 0.0);
      const bool ok = evaluateIK(pos, cell.stencil_quat[si], cfg.ee_link, ctx, eval_seeds, cfg.warm_start_mix,
                                 csr_trials, random_trials, timeout, solution);
      if (ok)
      {
        cell.stencil_label[si] = 1;
        cell.stencil_joint[si] = solution;
        ++pos_count;
        Seed local_seed;
        local_seed.quat = cell.stencil_quat[si];
        local_seed.joint = solution;
        eval_seeds.push_back(local_seed);
        if (si == 0)
        {
          cell.center_joint = solution;
          cell.has_center_joint = true;
        }
      }
      else
      {
        cell.stencil_label[si] = -1;
        cell.stencil_joint[si].clear();
        ++neg_count;
      }
    }

    if (pos_count == static_cast<int>(cell.stencil_label.size()))
    {
      cell.state = So3CellState::kInside;
    }
    else if (neg_count == static_cast<int>(cell.stencil_label.size()))
    {
      cell.state = So3CellState::kOutside;
    }
    else if (pos_count > 0 && neg_count > 0)
    {
      cell.state = So3CellState::kMixed;
    }
    else
    {
      cell.state = So3CellState::kUnknown;
    }
  };

  for (size_t ai = 0; ai < anchors.size(); ++ai)
  {
    if (ai < anchor_selected.size() && anchor_selected[ai] == 0)
    {
      anchor_start.push_back(writer.sample_count);
      cell_anchor_start.push_back(writer.cell_count);
      if (cfg.write_csr && cfg.stream_csr)
      {
        writer.appendAnchorStart(writer.sample_count);
        if (cfg.flush_per_anchor)
        {
          writer.flush();
        }
      }
      if (cfg.write_cells)
      {
        writer.appendCellAnchorStart(writer.cell_count);
      }
      continue;
    }

    const Anchor& anchor = anchors[ai];
    const Eigen::Vector3d pos(anchor.pos.x(), anchor.pos.y(), anchor.pos.z());

    std::vector<Seed> seeds = readSeeds(anchor.idx);
    if (seeds.size() < static_cast<size_t>(cfg.seeds_per_anchor))
    {
      const size_t missing = static_cast<size_t>(cfg.seeds_per_anchor) - seeds.size();
      auto extra = generateRandomSeeds(
          missing, toRosRngSeed(cfg.fallback_seed_base + static_cast<uint64_t>(ai) * cfg.fallback_seed_stride));
      seeds.insert(seeds.end(), extra.begin(), extra.end());
    }
    seeds = selectFarthestSeeds(seeds, cfg.seeds_per_anchor);
    if (seeds.empty())
    {
      RCLCPP_WARN(logger, "Anchor %zu has no seeds; skipping.", ai);
      anchor_start.push_back(writer.sample_count);
      cell_anchor_start.push_back(writer.cell_count);
      if (cfg.write_cells)
      {
        writer.appendCellAnchorStart(writer.cell_count);
      }
      continue;
    }

    std::vector<So3Cell> cells;
    cells.reserve(static_cast<size_t>(so3_base_cells) * 2);
    std::vector<int> current;
    current.reserve(base_quats.size());
    for (const auto& q : base_quats)
    {
      So3Cell cell;
      cell.parent = -1;
      cell.level = 0;
      cell.radius = base_radius;
      cell.center = canonicalizeQuat(q);
      cells.push_back(cell);
      current.push_back(static_cast<int>(cells.size() - 1));
    }

    std::vector<int> leaf_ids;
    for (int level = 0; level <= so3_max_depth && !current.empty(); ++level)
    {
      const int csr_trials = level == 0 ? cfg.ik_csr_trials_coarse : cfg.ik_csr_trials_refine;
      const int random_trials = level == 0 ? cfg.ik_random_trials_coarse : cfg.ik_random_trials_refine;
      const double timeout = level == 0 ? cfg.ik_timeout_coarse : cfg.ik_timeout_refine;
#pragma omp parallel for schedule(dynamic)
      for (int ci = 0; ci < static_cast<int>(current.size()); ++ci)
      {
        const int tid = 0
#ifdef _OPENMP
                        + omp_get_thread_num()
#endif
            ;
        auto& ctx = thread_contexts[static_cast<size_t>(tid)];
        certify_cell(cells[static_cast<size_t>(current[static_cast<size_t>(ci)])], pos, seeds, cells, ctx,
                     csr_trials, random_trials, timeout);
      }

      size_t level_inside = 0;
      size_t level_outside = 0;
      size_t level_mixed = 0;
      size_t level_unknown = 0;
      size_t refine_count = 0;
      for (int idx : current)
      {
        const So3Cell& cell = cells[static_cast<size_t>(idx)];
        if (cell.state == So3CellState::kInside)
        {
          ++level_inside;
        }
        else if (cell.state == So3CellState::kOutside)
        {
          ++level_outside;
        }
        else if (cell.state == So3CellState::kMixed)
        {
          ++level_mixed;
        }
        else
        {
          ++level_unknown;
        }
        const bool should_refine =
            (cell.state == So3CellState::kMixed || cell.state == So3CellState::kUnknown) &&
            cell.level < static_cast<uint8_t>(so3_max_depth) && cell.radius > cfg.so3_leaf_radius;
        if (should_refine)
        {
          ++refine_count;
        }
      }
      RCLCPP_INFO(logger,
                  "Anchor %zu level %d: current=%zu inside=%zu outside=%zu mixed=%zu unknown=%zu refine=%zu "
                  "leaf_so_far=%zu total_cells=%zu",
                  ai, level, current.size(), level_inside, level_outside, level_mixed, level_unknown, refine_count,
                  leaf_ids.size(), cells.size());

      std::vector<int> next;
      next.reserve(refine_count * 8);
      cells.reserve(cells.size() + refine_count * 8);
      for (int idx : current)
      {
        const So3Cell& cell = cells[static_cast<size_t>(idx)];
        const bool should_refine =
            (cell.state == So3CellState::kMixed || cell.state == So3CellState::kUnknown) &&
            cell.level < static_cast<uint8_t>(so3_max_depth) && cell.radius > cfg.so3_leaf_radius;
        if (!should_refine)
        {
          leaf_ids.push_back(idx);
          continue;
        }
        const uint8_t parent_level = cell.level;
        const double parent_radius = cell.radius;
        const Eigen::Quaterniond parent_center = cell.center;
        const double child_radius = 0.5 * parent_radius;
        const double offset = parent_radius / (2.0 * std::sqrt(3.0));
        for (int sx : { -1, 1 })
        {
          for (int sy : { -1, 1 })
          {
            for (int sz : { -1, 1 })
            {
              So3Cell child;
              child.parent = idx;
              child.level = static_cast<uint8_t>(parent_level + 1);
              child.radius = child_radius;
              child.center =
                  offsetQuat(parent_center, Eigen::Vector3d(static_cast<double>(sx) * offset,
                                                            static_cast<double>(sy) * offset,
                                                            static_cast<double>(sz) * offset));
              cells.push_back(child);
              next.push_back(static_cast<int>(cells.size() - 1));
            }
          }
        }
      }
      current.swap(next);
    }

    std::sort(leaf_ids.begin(), leaf_ids.end(), [&](int lhs, int rhs) {
      const So3Cell& a = cells[static_cast<size_t>(lhs)];
      const So3Cell& b = cells[static_cast<size_t>(rhs)];
      if (a.radius != b.radius)
      {
        return a.radius > b.radius;
      }
      return lhs < rhs;
    });

    size_t leaf_inside = 0;
    size_t leaf_outside = 0;
    size_t leaf_mixed = 0;
    size_t leaf_unknown = 0;
    for (const int leaf_id : leaf_ids)
    {
      const auto state = cells[static_cast<size_t>(leaf_id)].state;
      if (state == So3CellState::kInside)
      {
        ++leaf_inside;
      }
      else if (state == So3CellState::kOutside)
      {
        ++leaf_outside;
      }
      else if (state == So3CellState::kMixed)
      {
        ++leaf_mixed;
      }
      else
      {
        ++leaf_unknown;
      }
    }

    const bool full_reachable_candidate =
        !leaf_ids.empty() && leaf_outside == 0 && leaf_mixed == 0 && leaf_unknown == 0 && leaf_inside == leaf_ids.size();
    std::vector<BoundaryBracket> validation_brackets;
    bool full_reachable = full_reachable_candidate;
    if (full_reachable_candidate && cfg.full_reachable_validation_samples > 0)
    {
      const size_t probe_count = static_cast<size_t>(cfg.full_reachable_validation_samples);
      std::vector<Eigen::Quaterniond> validation_quats;
      validation_quats.reserve(probe_count);
      std::mt19937_64 rng(full_validation_seed_base + static_cast<uint64_t>(ai + 1) * 0x9e3779b97f4a7c15ULL);
      for (size_t i = 0; i < probe_count; ++i)
      {
        validation_quats.push_back(sampleUniformQuaternion(rng));
      }

      std::vector<uint8_t> validation_ok(probe_count, 0);
#pragma omp parallel for schedule(dynamic)
      for (int pi = 0; pi < static_cast<int>(probe_count); ++pi)
      {
        const int tid = 0
#ifdef _OPENMP
                        + omp_get_thread_num()
#endif
            ;
        auto& ctx = thread_contexts[static_cast<size_t>(tid)];
        std::vector<double> solution(joint_count, 0.0);
        const bool ok =
            evaluateIK(pos, validation_quats[static_cast<size_t>(pi)], cfg.ee_link, ctx, seeds, cfg.warm_start_mix,
                       cfg.ik_csr_trials_bisect, cfg.ik_random_trials_bisect, cfg.ik_timeout_bisect, solution);
        validation_ok[static_cast<size_t>(pi)] = ok ? 1 : 0;
      }

      size_t failed = 0;
      for (size_t pi = 0; pi < probe_count; ++pi)
      {
        if (validation_ok[pi] != 0)
        {
          continue;
        }
        ++failed;
        double best = std::numeric_limits<double>::infinity();
        int best_leaf = -1;
        for (const int leaf_id : leaf_ids)
        {
          const So3Cell& cell = cells[static_cast<size_t>(leaf_id)];
          if (!cell.has_center_joint)
          {
            continue;
          }
          const double d = geodesicDistance(cell.center, validation_quats[pi]);
          if (d < best)
          {
            best = d;
            best_leaf = leaf_id;
          }
        }
        if (best_leaf >= 0)
        {
          const So3Cell& pos_cell = cells[static_cast<size_t>(best_leaf)];
          BoundaryBracket bracket;
          bracket.q_pos = pos_cell.center;
          bracket.q_neg = validation_quats[pi];
          bracket.pos_joint = pos_cell.center_joint;
          validation_brackets.push_back(std::move(bracket));
        }
      }

      if (failed > 0)
      {
        full_reachable = false;
      }
      RCLCPP_INFO(logger,
                  "Anchor %zu full validation: probes=%zu failed=%zu validation_brackets=%zu candidate=%s final=%s",
                  ai, probe_count, failed, validation_brackets.size(), full_reachable_candidate ? "yes" : "no",
                  full_reachable ? "full" : "regular");
    }
    if (full_reachable)
    {
      anchor_full_reachable[ai] = 1;

      std::vector<CellOutput> cell_outputs;
      cell_outputs.reserve(leaf_ids.size());
      const size_t bucket_count = std::max<size_t>(size_t{ 1 }, q_ref.size());
      std::vector<std::vector<int>> full_buckets(bucket_count);
      for (const int leaf_id : leaf_ids)
      {
        const So3Cell& cell = cells[static_cast<size_t>(leaf_id)];
        CellOutput co;
        co.quat = canonicalizeQuat(cell.center);
        co.level = cell.level;
        co.state = static_cast<uint8_t>(cell.state);
        co.phi_graph = std::numeric_limits<float>::quiet_NaN();
        cell_outputs.push_back(co);
        full_buckets[q_ref_bucket(cell.center)].push_back(leaf_id);
      }

      std::vector<OutputSample> output_samples;
      output_samples.reserve(leaf_inside);
      std::vector<size_t> bucket_cursor(bucket_count, 0);
      size_t remaining = leaf_inside;
      while (remaining > 0)
      {
        bool progressed = false;
        for (size_t b = 0; b < bucket_count; ++b)
        {
          if (bucket_cursor[b] >= full_buckets[b].size())
          {
            continue;
          }
          const int leaf_id = full_buckets[b][bucket_cursor[b]++];
          const So3Cell& cell = cells[static_cast<size_t>(leaf_id)];
          OutputSample o;
          o.quat = canonicalizeQuat(cell.center);
          o.phi = static_cast<float>(cfg.phi_full);
          o.label = 1;
          o.method = 4;
          if (cell.has_center_joint)
          {
            o.joint.assign(cell.center_joint.begin(), cell.center_joint.end());
          }
          output_samples.push_back(std::move(o));
          --remaining;
          progressed = true;
        }
        if (!progressed)
        {
          break;
        }
      }

      if (!writer.appendCells(cell_outputs))
      {
        RCLCPP_ERROR(logger, "Failed to append cells for full-reachable anchor %zu", ai);
        writer.close();
        H5Fclose(base_file);
        rclcpp::shutdown();
        return 1;
      }
      if (!writer.appendSamples(output_samples))
      {
        RCLCPP_ERROR(logger, "Failed to append full-positive samples for anchor %zu", ai);
        writer.close();
        H5Fclose(base_file);
        rclcpp::shutdown();
        return 1;
      }

      anchor_start.push_back(writer.sample_count);
      cell_anchor_start.push_back(writer.cell_count);
      if (cfg.write_csr && cfg.stream_csr)
      {
        writer.appendAnchorStart(writer.sample_count);
        if (cfg.flush_per_anchor)
        {
          writer.flush();
        }
      }
      if (cfg.write_cells)
      {
        writer.appendCellAnchorStart(writer.cell_count);
      }
      RCLCPP_INFO(logger,
                  "Anchor %zu full-reachable: leaves=%zu method4=%zu phi_full=%.6f (no boundary/shell/global)",
                  ai, leaf_ids.size(), output_samples.size(), cfg.phi_full);
      continue;
    }

    const double graph_grid_cell = std::max(0.08, 4.0 * cfg.so3_leaf_radius);
    const double inv_graph_grid_cell = 1.0 / graph_grid_cell;
    auto grid_key = [&](const Eigen::Quaterniond& q) -> QuatGridKey {
      return { static_cast<int>(std::floor(q.x() * inv_graph_grid_cell)),
               static_cast<int>(std::floor(q.y() * inv_graph_grid_cell)),
               static_cast<int>(std::floor(q.z() * inv_graph_grid_cell)),
               static_cast<int>(std::floor(q.w() * inv_graph_grid_cell)) };
    };

    std::unordered_map<QuatGridKey, std::vector<int>, QuatGridKeyHash> graph_buckets;
    graph_buckets.reserve(leaf_ids.size() * 2);
    for (size_t i = 0; i < leaf_ids.size(); ++i)
    {
      const So3Cell& cell = cells[static_cast<size_t>(leaf_ids[i])];
      graph_buckets[grid_key(cell.center)].push_back(static_cast<int>(i));
    }

    std::vector<std::vector<std::pair<int, float>>> leaf_graph(leaf_ids.size());
    size_t graph_edges = 0;
    for (size_t i = 0; i < leaf_ids.size(); ++i)
    {
      const So3Cell& a = cells[static_cast<size_t>(leaf_ids[i])];
      const QuatGridKey base_key = grid_key(a.center);
      const double max_d = std::min(M_PI, 2.5 * a.radius);
      const double max_chord = std::sqrt(std::max(0.0, 2.0 - 2.0 * std::cos(0.5 * max_d)));
      const int range = std::max(1, static_cast<int>(std::ceil(max_chord * inv_graph_grid_cell)));
      for (int dx = -range; dx <= range; ++dx)
      {
        for (int dy = -range; dy <= range; ++dy)
        {
          for (int dz = -range; dz <= range; ++dz)
          {
            for (int dw = -range; dw <= range; ++dw)
            {
              const QuatGridKey key{ base_key.x + dx, base_key.y + dy, base_key.z + dz, base_key.w + dw };
              const auto bucket_it = graph_buckets.find(key);
              if (bucket_it == graph_buckets.end())
              {
                continue;
              }
              for (const int j_int : bucket_it->second)
              {
                const size_t j = static_cast<size_t>(j_int);
                if (j <= i)
                {
                  continue;
                }
                const So3Cell& b = cells[static_cast<size_t>(leaf_ids[j])];
                const double d = geodesicDistance(a.center, b.center);
                const double threshold = 1.25 * (a.radius + b.radius);
                if (d <= threshold)
                {
                  leaf_graph[i].push_back({ static_cast<int>(j), static_cast<float>(d) });
                  leaf_graph[j].push_back({ static_cast<int>(i), static_cast<float>(d) });
                  ++graph_edges;
                }
              }
            }
          }
        }
      }
    }
    RCLCPP_INFO(logger, "Anchor %zu graph: leaves=%zu edges=%zu grid_cell=%.4f", ai, leaf_ids.size(), graph_edges,
                graph_grid_cell);

    std::vector<OutputSample> output_samples;
    std::vector<CellOutput> cell_outputs;
    std::vector<Eigen::Quaterniond> boundary_quats;
    std::vector<BoundaryBracket> brackets = std::move(validation_brackets);
    std::vector<uint8_t> certified(leaf_ids.size(), 0);
    std::vector<uint8_t> boundary_source(leaf_ids.size(), 0);

    for (size_t i = 0; i < leaf_ids.size(); ++i)
    {
      const int sign_i = cell_sign(cells[static_cast<size_t>(leaf_ids[i])].state);
      certified[i] = sign_i == 0 ? 0 : 1;
      if (cells[static_cast<size_t>(leaf_ids[i])].state == So3CellState::kMixed)
      {
        double best = std::numeric_limits<double>::infinity();
        int best_pos = -1;
        int best_neg = -1;
        const auto& cell = cells[static_cast<size_t>(leaf_ids[i])];
        for (size_t a = 0; a < cell.stencil_label.size(); ++a)
        {
          if (cell.stencil_label[a] != 1)
          {
            continue;
          }
          for (size_t b = 0; b < cell.stencil_label.size(); ++b)
          {
            if (cell.stencil_label[b] != -1)
            {
              continue;
            }
            const double d = geodesicDistance(cell.stencil_quat[a], cell.stencil_quat[b]);
            if (d < best)
            {
              best = d;
              best_pos = static_cast<int>(a);
              best_neg = static_cast<int>(b);
            }
          }
        }
        if (best_pos >= 0 && best_neg >= 0)
        {
          BoundaryBracket bracket;
          bracket.q_pos = cell.stencil_quat[static_cast<size_t>(best_pos)];
          bracket.q_neg = cell.stencil_quat[static_cast<size_t>(best_neg)];
          bracket.pos_joint = cell.stencil_joint[static_cast<size_t>(best_pos)];
          brackets.push_back(bracket);
        }
      }
    }

    for (size_t i = 0; i < leaf_ids.size(); ++i)
    {
      const int sign_i = cell_sign(cells[static_cast<size_t>(leaf_ids[i])].state);
      for (const auto& [j_int, dist] : leaf_graph[i])
      {
        const size_t j = static_cast<size_t>(j_int);
        if (j <= i)
        {
          continue;
        }
        const int sign_j = cell_sign(cells[static_cast<size_t>(leaf_ids[j])].state);
        if (sign_i != 0 && sign_j != 0 && sign_i != sign_j)
        {
          const bool i_pos = sign_i > 0;
          const So3Cell& pos_cell = cells[static_cast<size_t>(leaf_ids[i_pos ? i : j])];
          const So3Cell& neg_cell = cells[static_cast<size_t>(leaf_ids[i_pos ? j : i])];
          if (pos_cell.has_center_joint)
          {
            BoundaryBracket bracket;
            bracket.q_pos = pos_cell.center;
            bracket.q_neg = neg_cell.center;
            bracket.pos_joint = pos_cell.center_joint;
            brackets.push_back(bracket);
          }
          boundary_source[i] = 1;
          boundary_source[j] = 1;
        }
        else if (sign_i != 0 || sign_j != 0)
        {
          const So3CellState state_i = cells[static_cast<size_t>(leaf_ids[i])].state;
          const So3CellState state_j = cells[static_cast<size_t>(leaf_ids[j])].state;
          if (state_i == So3CellState::kMixed || state_j == So3CellState::kMixed)
          {
            if (sign_i != 0)
            {
              boundary_source[i] = 1;
            }
            if (sign_j != 0)
            {
              boundary_source[j] = 1;
            }
          }
        }
      }
    }

    const size_t bracket_count_before_budget = brackets.size();
    if (cfg.max_boundary_brackets > 0 && brackets.size() > static_cast<size_t>(cfg.max_boundary_brackets))
    {
      const size_t limit = static_cast<size_t>(cfg.max_boundary_brackets);
      std::vector<BoundaryBracket> budgeted;
      budgeted.reserve(limit);
      for (size_t k = 0; k < limit; ++k)
      {
        const size_t idx =
            limit == 1 ? 0 : static_cast<size_t>((static_cast<unsigned long long>(k) * (brackets.size() - 1)) /
                                                 static_cast<unsigned long long>(limit - 1));
        budgeted.push_back(brackets[idx]);
      }
      brackets.swap(budgeted);
    }
    RCLCPP_INFO(logger, "Anchor %zu brackets: %zu%s refine_steps=%d", ai, bracket_count_before_budget,
                brackets.size() == bracket_count_before_budget ? "" : " (budgeted)", cfg.boundary_refine_steps);

    if (!brackets.empty())
    {
      std::vector<std::vector<OutputSample>> output_local(static_cast<size_t>(thread_count));
      std::vector<std::vector<Eigen::Quaterniond>> boundary_local(static_cast<size_t>(thread_count));
#pragma omp parallel for schedule(dynamic)
      for (int bi = 0; bi < static_cast<int>(brackets.size()); ++bi)
      {
        const int tid = 0
#ifdef _OPENMP
                        + omp_get_thread_num()
#endif
            ;
        auto& ctx = thread_contexts[static_cast<size_t>(tid)];
        auto& out = output_local[static_cast<size_t>(tid)];
        auto& bq = boundary_local[static_cast<size_t>(tid)];
        const BoundaryBracket& bracket = brackets[static_cast<size_t>(bi)];

        Eigen::Quaterniond q_pos_far = bracket.q_pos;
        Eigen::Quaterniond q_neg_far = bracket.q_neg;
        Eigen::Quaterniond q_pos = q_pos_far;
        Eigen::Quaterniond q_neg = q_neg_far;
        std::vector<double> q_pos_joint = bracket.pos_joint;
        std::vector<Seed> bisect_seeds = seeds;
        if (!q_pos_joint.empty())
        {
          Seed local_seed;
          local_seed.quat = q_pos;
          local_seed.joint = q_pos_joint;
          bisect_seeds.push_back(local_seed);
        }

        auto refine_once = [&]() {
          if (geodesicDistance(q_pos, q_neg) <= 1e-9)
          {
            return;
          }
          const Eigen::Quaterniond q_mid = slerpShortest(q_pos, q_neg, 0.5);
          std::vector<double> solution(joint_count, 0.0);
          const bool ok =
              evaluateIK(pos, q_mid, cfg.ee_link, ctx, bisect_seeds, cfg.warm_start_mix, cfg.ik_csr_trials_bisect,
                         cfg.ik_random_trials_bisect, cfg.ik_timeout_bisect, solution);
          if (ok)
          {
            q_pos = q_mid;
            q_pos_joint = solution;
            Seed local_seed;
            local_seed.quat = q_mid;
            local_seed.joint = solution;
            bisect_seeds.push_back(local_seed);
          }
          else
          {
            q_neg = q_mid;
          }
        };

        if (cfg.boundary_refine_steps >= 0)
        {
          for (int step = 0; step < cfg.boundary_refine_steps; ++step)
          {
            refine_once();
          }
        }
        else
        {
          while (geodesicDistance(q_pos, q_neg) > cfg.delta_boundary)
          {
            refine_once();
          }
        }

        const Eigen::Quaterniond q_star = slerpShortest(q_pos, q_neg, 0.5);
        bq.push_back(q_star);
        OutputSample boundary;
        boundary.quat = canonicalizeQuat(q_star);
        boundary.phi = 0.0f;
        boundary.label = 1;
        boundary.method = 2;
        if (!q_pos_joint.empty())
        {
          boundary.joint.assign(q_pos_joint.begin(), q_pos_joint.end());
        }
        out.push_back(boundary);

        const double d_pos = geodesicDistance(q_star, q_pos_far);
        const double d_neg = geodesicDistance(q_star, q_neg_far);
        for (double delta : cfg.shell_deltas)
        {
          if (d_pos > 1e-9 && delta <= d_pos)
          {
            const double t = delta / d_pos;
            OutputSample s;
            s.quat = canonicalizeQuat(slerpShortest(q_star, q_pos_far, t));
            s.phi = static_cast<float>(delta);
            s.label = 1;
            s.method = 0;
            if (!q_pos_joint.empty())
            {
              s.joint.assign(q_pos_joint.begin(), q_pos_joint.end());
            }
            out.push_back(s);
          }
          if (d_neg > 1e-9 && delta <= d_neg)
          {
            const double t = delta / d_neg;
            OutputSample s;
            s.quat = canonicalizeQuat(slerpShortest(q_star, q_neg_far, t));
            s.phi = static_cast<float>(-delta);
            s.label = -1;
            s.method = 1;
            out.push_back(s);
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

    size_t global_added = 0;
    for (size_t li = 0; li < leaf_ids.size(); ++li)
    {
      So3Cell& cell = cells[static_cast<size_t>(leaf_ids[li])];
      const int sign = cell_sign(cell.state);
      if (sign != 0 && !boundary_quats.empty())
      {
        double abs_phi = std::numeric_limits<double>::infinity();
        for (const auto& bq : boundary_quats)
        {
          abs_phi = std::min(abs_phi, geodesicDistance(cell.center, bq));
        }
        cell.phi_graph = static_cast<float>(static_cast<double>(sign) * abs_phi);
        OutputSample o;
        o.quat = canonicalizeQuat(cell.center);
        o.label = sign > 0 ? 1 : -1;
        o.method = 3;
        o.phi = cell.phi_graph;
        if (sign > 0 && cell.has_center_joint)
        {
          o.joint.assign(cell.center_joint.begin(), cell.center_joint.end());
        }
        output_samples.push_back(std::move(o));
        ++global_added;
      }

      CellOutput co;
      co.quat = canonicalizeQuat(cell.center);
      co.level = cell.level;
      co.state = static_cast<uint8_t>(cell.state);
      co.phi_graph = cell.phi_graph;
      cell_outputs.push_back(co);
    }

    if (!writer.appendCells(cell_outputs))
    {
      RCLCPP_ERROR(logger, "Failed to append cells for anchor %zu", ai);
      writer.close();
      H5Fclose(base_file);
      rclcpp::shutdown();
      return 1;
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
    cell_anchor_start.push_back(writer.cell_count);
    if (cfg.write_csr && cfg.stream_csr)
    {
      writer.appendAnchorStart(writer.sample_count);
      if (cfg.flush_per_anchor)
      {
        writer.flush();
      }
    }
    if (cfg.write_cells)
    {
      writer.appendCellAnchorStart(writer.cell_count);
    }
    RCLCPP_INFO(logger, "Anchor %zu done: cells=%zu boundary=%zu global=%zu samples=%zu", ai, leaf_ids.size(),
                boundary_quats.size(), global_added, output_samples.size());
  }

  writer.writeArray(writer.anchors_group, "full_reachable", H5T_STD_U8LE,
                    { anchor_full_reachable.size() }, anchor_full_reachable.data());

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
  if (cfg.write_cells)
  {
    std::vector<uint64_t> cell_ids(writer.cell_count, 0);
    std::iota(cell_ids.begin(), cell_ids.end(), 0);
    writer.writeArray(writer.cells_csr_group, "cell_index", H5T_STD_U64LE,
                      { static_cast<hsize_t>(cell_ids.size()) }, cell_ids.data());
  }

  writer.close();
  H5Fclose(base_file);
  RCLCPP_INFO(logger, "Done. Output: %s", cfg.output_path.c_str());
  std::fflush(stdout);
  std::fflush(stderr);
  std::_Exit(0);
}
