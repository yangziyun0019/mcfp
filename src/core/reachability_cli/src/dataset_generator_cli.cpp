// File: dataset_generator_cli.cpp
// Purpose: Generate position reachability datasets and voxel fields from robot URDF and SRDF inputs.
// Usage: ros2 run reachability_cli dataset_generator_cli --config <yaml> [--output <dir>]

#include "reachability_cli/model_loader.h"
#include "reachability_cli/npy_writer.h"

#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

#include <geometry_msgs/msg/pose.hpp>
#include <random_numbers/random_numbers.h>

#include <moveit/collision_detection/collision_common.h>
#include <moveit/kdl_kinematics_plugin/kdl_kinematics_plugin.h>
#include <moveit/robot_model/joint_model_group.h>
#include <moveit/robot_state/robot_state.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <hdf5.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

namespace
{
struct Options
{
  std::string config_path;
  std::string output_dir_override;
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

  double voxel_size = 0.003;
  double aabb_scale = 1.5;
  int aabb_samples = 20000;
  int fk_samples = 200000;
  int fk_max_attempts = 1000000;
  int random_seed = 0;
  int fk_log_interval = 10000;
  int thread_count = 0;
  int orientation_bins = 0;
  std::string coverage_mode = "heat";
  int coverage_check_interval = 10000;
  double coverage_delta_ratio = 0.001;
  int coverage_patience = 5;
  int coverage_min_occupied = 10000;
  double coverage_heat_min = 1000.0;
  double coverage_heat_alpha = 0.2;

  int ik_attempts = 200;
  double ik_timeout = 0.005;
  int ik_attempts_refine = 400;
  double ik_timeout_refine = 0.01;
  double search_discretization = 0.005;

  size_t hole_fill_max_voxels = 0;
  int closing_radius_voxels = 0;

  std::string output_dir;
  bool write_fk_samples = false;
  bool write_voxel_counts = true;
  bool write_sdf_grid = true;
  bool write_orientation_coverage = false;
  bool write_npy = true;
  bool write_metadata_yaml = true;
  bool write_hdf5 = false;
  std::string hdf5_path;
  size_t hdf5_chunk = 100000;
};

struct Grid
{
  Eigen::Vector3d origin;
  double voxel_size = 0.003;
  size_t nx = 0;
  size_t ny = 0;
  size_t nz = 0;

  size_t index(size_t x, size_t y, size_t z) const
  {
    return (x * ny + y) * nz + z;
  }

  Eigen::Vector3d center(size_t x, size_t y, size_t z) const
  {
    return origin +
           voxel_size * Eigen::Vector3d(static_cast<double>(x) + 0.5, static_cast<double>(y) + 0.5,
                                        static_cast<double>(z) + 0.5);
  }

  size_t size() const
  {
    return nx * ny * nz;
  }
};

struct SampleBuffer
{
  std::vector<float> positions;
  std::vector<float> quats;
  std::vector<float> joints;
  std::vector<uint64_t> voxels;

  void clear()
  {
    positions.clear();
    quats.clear();
    joints.clear();
    voxels.clear();
  }

  void reserve(size_t sample_count, size_t joint_count)
  {
    positions.reserve(sample_count * 3);
    quats.reserve(sample_count * 4);
    joints.reserve(sample_count * joint_count);
    voxels.reserve(sample_count);
  }

  size_t size() const
  {
    return voxels.size();
  }
};

enum class CellLabel : uint8_t
{
  kBoundary = 0,
  kInside = 1,
  kOutside = 2
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
    else if (arg == "--output" && i + 1 < argc)
    {
      options.output_dir_override = argv[++i];
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
  std::cerr << "Usage: dataset_generator_cli --config <path> [--output <dir>]\n";
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

Config loadConfig(const std::string& path)
{
  YAML::Node root = YAML::LoadFile(path);
  YAML::Node robot = root["robot"] ? root["robot"] : root;
  YAML::Node sampling = root["sampling"] ? root["sampling"] : root;
  YAML::Node ik = root["ik"] ? root["ik"] : root;
  YAML::Node sdf = root["sdf"] ? root["sdf"] : root;
  YAML::Node output = root["output"] ? root["output"] : root;

  Config cfg;
  cfg.robot_name = getScalar<std::string>(robot, "name", "robot");
  cfg.urdf_path = getRequiredScalar<std::string>(robot, "urdf");
  cfg.srdf_path = getRequiredScalar<std::string>(robot, "srdf");
  cfg.group_name = getRequiredScalar<std::string>(robot, "group_name");
  cfg.base_link = getRequiredScalar<std::string>(robot, "base_link");
  cfg.ee_link = getRequiredScalar<std::string>(robot, "ee_link");
  cfg.joint_names = getStringList(robot, "joint_names");

  cfg.voxel_size = getScalar<double>(sampling, "voxel_size", cfg.voxel_size);
  cfg.aabb_scale = getScalar<double>(sampling, "aabb_scale", cfg.aabb_scale);
  cfg.aabb_samples = getScalar<int>(sampling, "aabb_samples", cfg.aabb_samples);
  cfg.fk_samples = getScalar<int>(sampling, "fk_samples", cfg.fk_samples);
  cfg.fk_max_attempts = getScalar<int>(sampling, "fk_max_attempts", cfg.fk_max_attempts);
  cfg.random_seed = getScalar<int>(sampling, "random_seed", cfg.random_seed);
  cfg.fk_log_interval = getScalar<int>(sampling, "fk_log_interval", cfg.fk_log_interval);
  cfg.thread_count = getScalar<int>(sampling, "threads", cfg.thread_count);
  cfg.orientation_bins = getScalar<int>(sampling, "orientation_bins", cfg.orientation_bins);
  cfg.coverage_mode = getScalar<std::string>(sampling, "coverage_mode", cfg.coverage_mode);
  cfg.coverage_check_interval = getScalar<int>(sampling, "coverage_check_interval", cfg.coverage_check_interval);
  cfg.coverage_delta_ratio = getScalar<double>(sampling, "coverage_delta_ratio", cfg.coverage_delta_ratio);
  cfg.coverage_patience = getScalar<int>(sampling, "coverage_patience", cfg.coverage_patience);
  cfg.coverage_min_occupied = getScalar<int>(sampling, "coverage_min_occupied", cfg.coverage_min_occupied);
  cfg.coverage_heat_min = getScalar<double>(sampling, "coverage_heat_min", cfg.coverage_heat_min);
  cfg.coverage_heat_alpha = getScalar<double>(sampling, "coverage_heat_alpha", cfg.coverage_heat_alpha);

  cfg.ik_attempts = getScalar<int>(ik, "attempts", cfg.ik_attempts);
  cfg.ik_timeout = getScalar<double>(ik, "timeout", cfg.ik_timeout);
  cfg.ik_attempts_refine = getScalar<int>(ik, "attempts_refine", cfg.ik_attempts_refine);
  cfg.ik_timeout_refine = getScalar<double>(ik, "timeout_refine", cfg.ik_timeout_refine);
  cfg.search_discretization = getScalar<double>(ik, "search_discretization", cfg.search_discretization);

  cfg.closing_radius_voxels = getScalar<int>(sdf, "closing_radius_voxels", cfg.closing_radius_voxels);

  cfg.output_dir = getRequiredScalar<std::string>(output, "dir");
  cfg.write_fk_samples = getScalar<bool>(output, "write_fk_samples", cfg.write_fk_samples);
  cfg.write_voxel_counts = getScalar<bool>(output, "write_voxel_counts", cfg.write_voxel_counts);
  cfg.write_sdf_grid = getScalar<bool>(output, "write_sdf_grid", cfg.write_sdf_grid);
  cfg.write_orientation_coverage = getScalar<bool>(output, "write_orientation_coverage", cfg.write_orientation_coverage);
  cfg.write_npy = getScalar<bool>(output, "write_npy", cfg.write_npy);
  cfg.write_metadata_yaml = getScalar<bool>(output, "write_metadata_yaml", cfg.write_metadata_yaml);
  cfg.write_hdf5 = getScalar<bool>(output, "write_hdf5", cfg.write_hdf5);
  cfg.hdf5_path = getScalar<std::string>(output, "hdf5_path", cfg.hdf5_path);
  cfg.hdf5_chunk = getScalar<size_t>(output, "hdf5_chunk", cfg.hdf5_chunk);

  cfg.hole_fill_max_voxels = getScalar<size_t>(sdf, "hole_fill_max_voxels", cfg.hole_fill_max_voxels);

  return cfg;
}

bool ensureDirectory(const std::string& path)
{
  if (path.empty())
  {
    return false;
  }

  std::string current;
  if (path[0] == '/')
  {
    current = "/";
  }

  std::istringstream iss(path);
  std::string token;
  while (std::getline(iss, token, '/'))
  {
    if (token.empty())
    {
      continue;
    }
    if (current.size() > 1 && current.back() != '/')
    {
      current += "/";
    }
    current += token;
    struct stat st;
    if (stat(current.c_str(), &st) != 0)
    {
      if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST)
      {
        return false;
      }
    }
    else if (!S_ISDIR(st.st_mode))
    {
      return false;
    }
  }
  return true;
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

  Eigen::Quaterniond q(w, x, y, z);
  if (q.w() < 0.0)
  {
    q.coeffs() *= -1.0;
  }
  return q;
}

bool feasiblePos(const Eigen::Vector3d& position, moveit::core::RobotState& state,
                 const moveit::core::JointModelGroup* jmg, const std::string& ee_link, double ik_timeout,
                 int ik_attempts, planning_scene::PlanningScenePtr scene,
                 collision_detection::CollisionRequest& request, collision_detection::CollisionResult& result,
                 random_numbers::RandomNumberGenerator& rng)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = position.x();
  pose.position.y = position.y();
  pose.position.z = position.z();

  for (int attempt = 0; attempt < ik_attempts; ++attempt)
  {
    const Eigen::Quaterniond q = sampleUniformQuaternion(rng);
    pose.orientation.w = q.w();
    pose.orientation.x = q.x();
    pose.orientation.y = q.y();
    pose.orientation.z = q.z();

    state.setToRandomPositions(jmg, rng);
    const bool ik_ok = state.setFromIK(jmg, pose, ee_link, ik_timeout);
    if (!ik_ok)
    {
      continue;
    }

    state.update();
    if (!state.satisfiesBounds(jmg))
    {
      continue;
    }

    result.clear();
    scene->checkSelfCollision(request, result, state);
    if (!result.collision)
    {
      return true;
    }
  }
  return false;
}

void distanceTransform1D(const std::vector<float>& f, int n, std::vector<float>& d, std::vector<int>& v,
                         std::vector<float>& z)
{
  int k = 0;
  v[0] = 0;
  z[0] = -1e20f;
  z[1] = 1e20f;

  for (int q = 1; q < n; ++q)
  {
    float s = ((f[q] + static_cast<float>(q * q)) - (f[v[k]] + static_cast<float>(v[k] * v[k]))) /
              (2.0f * static_cast<float>(q - v[k]));
    while (s <= z[k])
    {
      --k;
      s = ((f[q] + static_cast<float>(q * q)) - (f[v[k]] + static_cast<float>(v[k] * v[k]))) /
          (2.0f * static_cast<float>(q - v[k]));
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = 1e20f;
  }

  k = 0;
  for (int q = 0; q < n; ++q)
  {
    while (z[k + 1] < static_cast<float>(q))
    {
      ++k;
    }
    const float diff = static_cast<float>(q - v[k]);
    d[q] = diff * diff + f[v[k]];
  }
}

void distanceTransform3D(std::vector<float>& grid, const Grid& g)
{
  const size_t max_dim = std::max(g.nx, std::max(g.ny, g.nz));
  std::vector<float> line(max_dim, 0.0f);
  std::vector<float> line_out(max_dim, 0.0f);
  std::vector<int> v(max_dim, 0);
  std::vector<float> z(max_dim + 1, 0.0f);

  for (size_t x = 0; x < g.nx; ++x)
  {
    for (size_t y = 0; y < g.ny; ++y)
    {
      for (size_t z_idx = 0; z_idx < g.nz; ++z_idx)
      {
        line[z_idx] = grid[g.index(x, y, z_idx)];
      }
      distanceTransform1D(line, static_cast<int>(g.nz), line_out, v, z);
      for (size_t z_idx = 0; z_idx < g.nz; ++z_idx)
      {
        grid[g.index(x, y, z_idx)] = line_out[z_idx];
      }
    }
  }

  for (size_t x = 0; x < g.nx; ++x)
  {
    for (size_t z_idx = 0; z_idx < g.nz; ++z_idx)
    {
      for (size_t y = 0; y < g.ny; ++y)
      {
        line[y] = grid[g.index(x, y, z_idx)];
      }
      distanceTransform1D(line, static_cast<int>(g.ny), line_out, v, z);
      for (size_t y = 0; y < g.ny; ++y)
      {
        grid[g.index(x, y, z_idx)] = line_out[y];
      }
    }
  }

  for (size_t y = 0; y < g.ny; ++y)
  {
    for (size_t z_idx = 0; z_idx < g.nz; ++z_idx)
    {
      for (size_t x = 0; x < g.nx; ++x)
      {
        line[x] = grid[g.index(x, y, z_idx)];
      }
      distanceTransform1D(line, static_cast<int>(g.nx), line_out, v, z);
      for (size_t x = 0; x < g.nx; ++x)
      {
        grid[g.index(x, y, z_idx)] = line_out[x];
      }
    }
  }
}

size_t fillSmallHoles(std::vector<uint8_t>& labels, const Grid& g, size_t max_size, rclcpp::Logger logger)
{
  if (max_size == 0)
  {
    return 0;
  }

  const size_t total = g.size();
  std::vector<uint8_t> visited(total, 0);
  std::vector<size_t> queue;
  queue.reserve(1024);

  auto isOutside = [&](size_t idx) {
    return labels[idx] == static_cast<uint8_t>(CellLabel::kOutside);
  };

  auto bfs_mark = [&](size_t start) {
    queue.clear();
    queue.push_back(start);
    visited[start] = 1;
    for (size_t qi = 0; qi < queue.size(); ++qi)
    {
      const size_t idx = queue[qi];
      const size_t x = idx / (g.ny * g.nz);
      const size_t rem = idx % (g.ny * g.nz);
      const size_t y = rem / g.nz;
      const size_t z = rem % g.nz;

      auto try_add = [&](size_t nidx) {
        if (!visited[nidx] && isOutside(nidx))
        {
          visited[nidx] = 1;
          queue.push_back(nidx);
        }
      };

      if (x > 0)
      {
        try_add(idx - g.ny * g.nz);
      }
      if (x + 1 < g.nx)
      {
        try_add(idx + g.ny * g.nz);
      }
      if (y > 0)
      {
        try_add(idx - g.nz);
      }
      if (y + 1 < g.ny)
      {
        try_add(idx + g.nz);
      }
      if (z > 0)
      {
        try_add(idx - 1);
      }
      if (z + 1 < g.nz)
      {
        try_add(idx + 1);
      }
    }
  };

  auto bfs_collect = [&](size_t start, std::vector<size_t>& component) {
    queue.clear();
    component.clear();
    queue.push_back(start);
    visited[start] = 1;
    component.push_back(start);
    for (size_t qi = 0; qi < queue.size(); ++qi)
    {
      const size_t idx = queue[qi];
      const size_t x = idx / (g.ny * g.nz);
      const size_t rem = idx % (g.ny * g.nz);
      const size_t y = rem / g.nz;
      const size_t z = rem % g.nz;

      auto try_add = [&](size_t nidx) {
        if (!visited[nidx] && isOutside(nidx))
        {
          visited[nidx] = 1;
          queue.push_back(nidx);
          component.push_back(nidx);
        }
      };

      if (x > 0)
      {
        try_add(idx - g.ny * g.nz);
      }
      if (x + 1 < g.nx)
      {
        try_add(idx + g.ny * g.nz);
      }
      if (y > 0)
      {
        try_add(idx - g.nz);
      }
      if (y + 1 < g.ny)
      {
        try_add(idx + g.nz);
      }
      if (z > 0)
      {
        try_add(idx - 1);
      }
      if (z + 1 < g.nz)
      {
        try_add(idx + 1);
      }
    }
  };

  auto seed_boundary = [&](size_t x, size_t y, size_t z) {
    const size_t idx = g.index(x, y, z);
    if (!visited[idx] && isOutside(idx))
    {
      bfs_mark(idx);
    }
  };

  for (size_t y = 0; y < g.ny; ++y)
  {
    for (size_t z = 0; z < g.nz; ++z)
    {
      seed_boundary(0, y, z);
      seed_boundary(g.nx - 1, y, z);
    }
  }
  for (size_t x = 0; x < g.nx; ++x)
  {
    for (size_t z = 0; z < g.nz; ++z)
    {
      seed_boundary(x, 0, z);
      seed_boundary(x, g.ny - 1, z);
    }
  }
  for (size_t x = 0; x < g.nx; ++x)
  {
    for (size_t y = 0; y < g.ny; ++y)
    {
      seed_boundary(x, y, 0);
      seed_boundary(x, y, g.nz - 1);
    }
  }

  size_t filled = 0;
  std::vector<size_t> component;
  component.reserve(std::min<size_t>(max_size, 10000));

  for (size_t idx = 0; idx < total; ++idx)
  {
    if (isOutside(idx) && !visited[idx])
    {
      bfs_collect(idx, component);
      if (component.size() <= max_size)
      {
        for (size_t cidx : component)
        {
          labels[cidx] = static_cast<uint8_t>(CellLabel::kInside);
        }
        filled += component.size();
      }
    }
  }

  if (filled > 0)
  {
    RCLCPP_INFO(logger, "Hole fill: filled %zu voxels (max_size=%zu)", filled, max_size);
  }
  else
  {
    RCLCPP_INFO(logger, "Hole fill: no holes filled (max_size=%zu)", max_size);
  }

  return filled;
}

size_t applyClosing(std::vector<uint8_t>& labels, const Grid& g, int radius, rclcpp::Logger logger)
{
  if (radius <= 0)
  {
    return 0;
  }

  const size_t total = g.size();
  std::vector<uint8_t> inside(total, 0);
  size_t inside_before = 0;
  for (size_t i = 0; i < total; ++i)
  {
    if (labels[i] == static_cast<uint8_t>(CellLabel::kInside))
    {
      inside[i] = 1;
      ++inside_before;
    }
  }

  if (inside_before == 0)
  {
    RCLCPP_WARN(logger, "Closing: no inside voxels, skip (radius=%d)", radius);
    return 0;
  }

  std::vector<uint8_t> dilated(total, 0);

  const int nx = static_cast<int>(g.nx);
  const int ny = static_cast<int>(g.ny);
  const int nz = static_cast<int>(g.nz);
  const int r = radius;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int x = 0; x < nx; ++x)
  {
    for (int y = 0; y < ny; ++y)
    {
      for (int z = 0; z < nz; ++z)
      {
        const size_t idx = g.index(static_cast<size_t>(x), static_cast<size_t>(y), static_cast<size_t>(z));
        bool any = false;
        const int x0 = std::max(0, x - r);
        const int x1 = std::min(nx - 1, x + r);
        const int y0 = std::max(0, y - r);
        const int y1 = std::min(ny - 1, y + r);
        const int z0 = std::max(0, z - r);
        const int z1 = std::min(nz - 1, z + r);
        for (int xi = x0; xi <= x1 && !any; ++xi)
        {
          for (int yi = y0; yi <= y1 && !any; ++yi)
          {
            for (int zi = z0; zi <= z1; ++zi)
            {
              const size_t nidx =
                  g.index(static_cast<size_t>(xi), static_cast<size_t>(yi), static_cast<size_t>(zi));
              if (inside[nidx])
              {
                any = true;
                break;
              }
            }
          }
        }
        dilated[idx] = any ? 1 : 0;
      }
    }
  }

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int x = 0; x < nx; ++x)
  {
    for (int y = 0; y < ny; ++y)
    {
      for (int z = 0; z < nz; ++z)
      {
        const size_t idx = g.index(static_cast<size_t>(x), static_cast<size_t>(y), static_cast<size_t>(z));
        bool all = true;
        if (x < r || x + r >= nx || y < r || y + r >= ny || z < r || z + r >= nz)
        {
          all = false;
        }
        else
        {
          for (int xi = x - r; xi <= x + r && all; ++xi)
          {
            for (int yi = y - r; yi <= y + r && all; ++yi)
            {
              for (int zi = z - r; zi <= z + r; ++zi)
              {
                const size_t nidx =
                    g.index(static_cast<size_t>(xi), static_cast<size_t>(yi), static_cast<size_t>(zi));
                if (!dilated[nidx])
                {
                  all = false;
                  break;
                }
              }
            }
          }
        }
        inside[idx] = all ? 1 : 0;
      }
    }
  }

  size_t inside_after = 0;
  for (size_t i = 0; i < total; ++i)
  {
    if (inside[i])
    {
      labels[i] = static_cast<uint8_t>(CellLabel::kInside);
      ++inside_after;
    }
    else
    {
      labels[i] = static_cast<uint8_t>(CellLabel::kOutside);
    }
  }

  const long delta = static_cast<long>(inside_after) - static_cast<long>(inside_before);
  RCLCPP_INFO(logger, "Closing: radius=%d inside_before=%zu inside_after=%zu delta=%ld", radius, inside_before,
              inside_after, delta);
  return inside_after;
}

std::string buildMetadataYaml(const Config& cfg, const Grid& g, const Eigen::Vector3d& aabb_min,
                              const Eigen::Vector3d& aabb_max, size_t fk_kept, size_t fk_attempts,
                              size_t fk_stored, size_t inside_count, size_t outside_count, size_t boundary_count,
                              int seed_used)
{
  std::ostringstream out;
  out << "robot_name: " << cfg.robot_name << "\n";
  out << "urdf: " << cfg.urdf_path << "\n";
  out << "srdf: " << cfg.srdf_path << "\n";
  out << "group_name: " << cfg.group_name << "\n";
  out << "base_link: " << cfg.base_link << "\n";
  out << "ee_link: " << cfg.ee_link << "\n";
  out << "voxel_size: " << cfg.voxel_size << "\n";
  out << "aabb_scale: " << cfg.aabb_scale << "\n";
  out << "aabb_min: [" << aabb_min.x() << ", " << aabb_min.y() << ", " << aabb_min.z() << "]\n";
  out << "aabb_max: [" << aabb_max.x() << ", " << aabb_max.y() << ", " << aabb_max.z() << "]\n";
  out << "grid_dims: [" << g.nx << ", " << g.ny << ", " << g.nz << "]\n";
  out << "grid_origin: [" << g.origin.x() << ", " << g.origin.y() << ", " << g.origin.z() << "]\n";
  out << "random_seed: " << seed_used << "\n";
  out << "aabb_samples: " << cfg.aabb_samples << "\n";
  out << "fk_samples_target: " << cfg.fk_samples << "\n";
  out << "fk_samples_kept: " << fk_kept << "\n";
  out << "fk_samples_attempts: " << fk_attempts << "\n";
  out << "fk_samples_stored: " << fk_stored << "\n";
  out << "orientation_bins: " << cfg.orientation_bins << "\n";
  out << "write_orientation_coverage: " << (cfg.write_orientation_coverage ? "true" : "false") << "\n";
  out << "coverage_mode: " << cfg.coverage_mode << "\n";
  out << "coverage_check_interval: " << cfg.coverage_check_interval << "\n";
  out << "coverage_delta_ratio: " << cfg.coverage_delta_ratio << "\n";
  out << "coverage_patience: " << cfg.coverage_patience << "\n";
  out << "coverage_min_occupied: " << cfg.coverage_min_occupied << "\n";
  out << "coverage_heat_min: " << cfg.coverage_heat_min << "\n";
  out << "coverage_heat_alpha: " << cfg.coverage_heat_alpha << "\n";
  out << "search_discretization: " << cfg.search_discretization << "\n";
  out << "inside_count: " << inside_count << "\n";
  out << "outside_count: " << outside_count << "\n";
  out << "boundary_count: " << boundary_count << "\n";
  out << "hole_fill_max_voxels: " << cfg.hole_fill_max_voxels << "\n";
  out << "closing_radius_voxels: " << cfg.closing_radius_voxels << "\n";
  out << "write_hdf5: " << (cfg.write_hdf5 ? "true" : "false") << "\n";
  out << "hdf5_path: " << cfg.hdf5_path << "\n";
  out << "hdf5_chunk: " << cfg.hdf5_chunk << "\n";
  out << "write_npy: " << (cfg.write_npy ? "true" : "false") << "\n";
  out << "write_metadata_yaml: " << (cfg.write_metadata_yaml ? "true" : "false") << "\n";
  return out.str();
}

void writeMetadata(const std::string& path, const Config& cfg, const Grid& g, const Eigen::Vector3d& aabb_min,
                   const Eigen::Vector3d& aabb_max, size_t fk_kept, size_t fk_attempts, size_t fk_stored,
                   size_t inside_count, size_t outside_count, size_t boundary_count, int seed_used)
{
  std::ofstream out(path);
  if (!out)
  {
    return;
  }
  out << buildMetadataYaml(cfg, g, aabb_min, aabb_max, fk_kept, fk_attempts, fk_stored, inside_count, outside_count,
                           boundary_count, seed_used);
}

std::string readFileToString(const std::string& path)
{
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in)
  {
    return std::string();
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

struct Hdf5Writer
{
  hid_t file = -1;
  hid_t samples_group = -1;
  hid_t grid_group = -1;
  hid_t meta_group = -1;
  hid_t csr_group = -1;
  hid_t pos_dset = -1;
  hid_t quat_dset = -1;
  hid_t joint_dset = -1;
  hid_t voxel_dset = -1;
  size_t sample_count = 0;
  size_t joint_count = 0;
  size_t chunk_rows = 0;

  bool open(const std::string& path, size_t chunk, size_t joint_dim)
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

    samples_group = H5Gcreate2(file, "/samples", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    grid_group = H5Gcreate2(file, "/grid", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    meta_group = H5Gcreate2(file, "/meta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    csr_group = H5Gcreate2(file, "/csr", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (samples_group < 0 || grid_group < 0 || meta_group < 0 || csr_group < 0)
    {
      return false;
    }

    chunk_rows = chunk;
    joint_count = joint_dim;

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

    pos_dset = create2d("pos", H5T_IEEE_F32LE, 3);
    quat_dset = create2d("quat", H5T_IEEE_F32LE, 4);
    joint_dset = create2d("joint", H5T_IEEE_F32LE, static_cast<hsize_t>(joint_count));
    voxel_dset = create1d("voxel", H5T_STD_U64LE);

    if (pos_dset < 0 || quat_dset < 0 || joint_dset < 0 || voxel_dset < 0)
    {
      return false;
    }
    return true;
  }

  bool appendSamples(const SampleBuffer& buffer)
  {
    const size_t count = buffer.size();
    if (count == 0)
    {
      return true;
    }
    if (buffer.positions.size() != count * 3 || buffer.quats.size() != count * 4 ||
        buffer.joints.size() != count * joint_count)
    {
      return false;
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

    if (!append2d(pos_dset, H5T_IEEE_F32LE, buffer.positions.data(), 3))
    {
      return false;
    }
    if (!append2d(quat_dset, H5T_IEEE_F32LE, buffer.quats.data(), 4))
    {
      return false;
    }
    if (!append2d(joint_dset, H5T_IEEE_F32LE, buffer.joints.data(), static_cast<hsize_t>(joint_count)))
    {
      return false;
    }
    if (!append1d(voxel_dset, H5T_STD_U64LE, buffer.voxels.data()))
    {
      return false;
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

  bool writeStringArray(hid_t group, const std::string& name, const std::vector<std::string>& values)
  {
    if (group < 0)
    {
      return false;
    }
    std::vector<const char*> c_strs;
    c_strs.reserve(values.size());
    for (const auto& v : values)
    {
      c_strs.push_back(v.c_str());
    }
    hid_t dtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(dtype, H5T_VARIABLE);
    hsize_t dims[1] = { static_cast<hsize_t>(values.size()) };
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dset = H5Dcreate2(group, name.c_str(), dtype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const bool ok = dset >= 0 && H5Dwrite(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, c_strs.data()) >= 0;
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
    if (pos_dset >= 0)
    {
      H5Dclose(pos_dset);
    }
    if (quat_dset >= 0)
    {
      H5Dclose(quat_dset);
    }
    if (joint_dset >= 0)
    {
      H5Dclose(joint_dset);
    }
    if (voxel_dset >= 0)
    {
      H5Dclose(voxel_dset);
    }
    if (samples_group >= 0)
    {
      H5Gclose(samples_group);
    }
    if (grid_group >= 0)
    {
      H5Gclose(grid_group);
    }
    if (meta_group >= 0)
    {
      H5Gclose(meta_group);
    }
    if (csr_group >= 0)
    {
      H5Gclose(csr_group);
    }
    if (file >= 0)
    {
      H5Fclose(file);
    }
    file = -1;
  }
};

bool writeCsrToHdf5(Hdf5Writer& writer, const Grid& grid, const std::vector<uint32_t>& voxel_counts,
                    rclcpp::Logger logger)
{
  const size_t sample_count = writer.sample_count;
  if (sample_count == 0)
  {
    RCLCPP_WARN(logger, "CSR skipped: no samples stored.");
    return true;
  }

  const size_t grid_size = grid.size();
  if (voxel_counts.size() != grid_size)
  {
    RCLCPP_ERROR(logger, "CSR failed: voxel_counts size mismatch.");
    return false;
  }
  std::vector<uint64_t> voxel_start(grid_size + 1, 0);
  for (size_t i = 0; i < grid_size; ++i)
  {
    voxel_start[i + 1] = voxel_start[i] + static_cast<uint64_t>(voxel_counts[i]);
  }
  const size_t total_counts = static_cast<size_t>(voxel_start.back());
  if (total_counts != sample_count)
  {
    RCLCPP_WARN(logger, "CSR count mismatch: voxel_counts=%zu samples=%zu", total_counts, sample_count);
    if (total_counts < sample_count)
    {
      RCLCPP_ERROR(logger, "CSR aborted: sample count exceeds voxel counts.");
      return false;
    }
  }

  const bool use_u32 = sample_count <= static_cast<size_t>(std::numeric_limits<uint32_t>::max());
  const size_t index_bytes = sample_count * (use_u32 ? sizeof(uint32_t) : sizeof(uint64_t));
  RCLCPP_INFO(logger, "CSR sample_index buffer: %.2f MB", index_bytes / (1024.0 * 1024.0));

  std::vector<uint64_t> cursor(grid_size, 0);
  for (size_t i = 0; i < grid_size; ++i)
  {
    cursor[i] = voxel_start[i];
  }

  if (use_u32)
  {
    std::vector<uint32_t> sample_index(sample_count, 0);
    hid_t dset = H5Dopen2(writer.file, "/samples/voxel", H5P_DEFAULT);
    if (dset < 0)
    {
      RCLCPP_ERROR(logger, "CSR failed: cannot open /samples/voxel");
      return false;
    }
    const size_t chunk = std::max<size_t>(1, writer.chunk_rows);
    std::vector<uint64_t> voxel_chunk(chunk, 0);
    size_t offset = 0;
    size_t global_id = 0;
    while (offset < sample_count)
    {
      const size_t count = std::min(chunk, sample_count - offset);
      hsize_t start[1] = { static_cast<hsize_t>(offset) };
      hsize_t block[1] = { static_cast<hsize_t>(count) };
      hid_t filespace = H5Dget_space(dset);
      H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, block, nullptr);
      hid_t memspace = H5Screate_simple(1, block, nullptr);
      const bool ok =
          H5Dread(dset, H5T_STD_U64LE, memspace, filespace, H5P_DEFAULT, voxel_chunk.data()) >= 0;
      H5Sclose(memspace);
      H5Sclose(filespace);
      if (!ok)
      {
        H5Dclose(dset);
        RCLCPP_ERROR(logger, "CSR failed: cannot read /samples/voxel");
        return false;
      }

      for (size_t i = 0; i < count; ++i)
      {
        const uint64_t voxel_id = voxel_chunk[i];
        if (voxel_id >= grid_size)
        {
          continue;
        }
        const uint64_t pos = cursor[voxel_id]++;
        if (pos < sample_index.size())
        {
          sample_index[pos] = static_cast<uint32_t>(global_id);
        }
        ++global_id;
      }

      offset += count;
    }
    H5Dclose(dset);

    writer.writeArray(writer.csr_group, "voxel_start", H5T_STD_U64LE,
                      { static_cast<hsize_t>(voxel_start.size()) }, voxel_start.data());
    writer.writeArray(writer.csr_group, "sample_index", H5T_STD_U32LE,
                      { static_cast<hsize_t>(sample_index.size()) }, sample_index.data());
    writer.writeString(writer.csr_group, "index_dtype", "uint32");
  }
  else
  {
    std::vector<uint64_t> sample_index(sample_count, 0);
    hid_t dset = H5Dopen2(writer.file, "/samples/voxel", H5P_DEFAULT);
    if (dset < 0)
    {
      RCLCPP_ERROR(logger, "CSR failed: cannot open /samples/voxel");
      return false;
    }
    const size_t chunk = std::max<size_t>(1, writer.chunk_rows);
    std::vector<uint64_t> voxel_chunk(chunk, 0);
    size_t offset = 0;
    size_t global_id = 0;
    while (offset < sample_count)
    {
      const size_t count = std::min(chunk, sample_count - offset);
      hsize_t start[1] = { static_cast<hsize_t>(offset) };
      hsize_t block[1] = { static_cast<hsize_t>(count) };
      hid_t filespace = H5Dget_space(dset);
      H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, block, nullptr);
      hid_t memspace = H5Screate_simple(1, block, nullptr);
      const bool ok =
          H5Dread(dset, H5T_STD_U64LE, memspace, filespace, H5P_DEFAULT, voxel_chunk.data()) >= 0;
      H5Sclose(memspace);
      H5Sclose(filespace);
      if (!ok)
      {
        H5Dclose(dset);
        RCLCPP_ERROR(logger, "CSR failed: cannot read /samples/voxel");
        return false;
      }

      for (size_t i = 0; i < count; ++i)
      {
        const uint64_t voxel_id = voxel_chunk[i];
        if (voxel_id >= grid_size)
        {
          continue;
        }
        const uint64_t pos = cursor[voxel_id]++;
        if (pos < sample_index.size())
        {
          sample_index[pos] = static_cast<uint64_t>(global_id);
        }
        ++global_id;
      }

      offset += count;
    }
    H5Dclose(dset);

    writer.writeArray(writer.csr_group, "voxel_start", H5T_STD_U64LE,
                      { static_cast<hsize_t>(voxel_start.size()) }, voxel_start.data());
    writer.writeArray(writer.csr_group, "sample_index", H5T_STD_U64LE,
                      { static_cast<hsize_t>(sample_index.size()) }, sample_index.data());
    writer.writeString(writer.csr_group, "index_dtype", "uint64");
  }

  return true;
}

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

  auto node = rclcpp::Node::make_shared("dataset_generator_cli");
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
  const std::string config_yaml = readFileToString(options.config_path);

  if (!options.output_dir_override.empty())
  {
    cfg.output_dir = options.output_dir_override;
  }

  if (cfg.write_hdf5)
  {
    if (cfg.hdf5_path.empty())
    {
      cfg.hdf5_path = cfg.output_dir + "/dataset.h5";
    }
  }

  int thread_count = cfg.thread_count;
#ifdef _OPENMP
  if (thread_count <= 0)
  {
    thread_count = omp_get_max_threads();
  }
  omp_set_num_threads(thread_count);
#else
  if (thread_count > 1)
  {
    RCLCPP_WARN(logger, "OpenMP not available, forcing sampling.threads=1");
  }
  thread_count = 1;
#endif

  if (cfg.coverage_mode != "heat" && cfg.coverage_mode != "ratio")
  {
    RCLCPP_WARN(logger, "Unknown sampling.coverage_mode='%s', fallback to 'heat'", cfg.coverage_mode.c_str());
    cfg.coverage_mode = "heat";
  }
  if (cfg.coverage_heat_min < 0.0)
  {
    cfg.coverage_heat_min = 0.0;
  }

  RCLCPP_INFO(logger, "Config: robot=%s group=%s ee_link=%s", cfg.robot_name.c_str(), cfg.group_name.c_str(),
              cfg.ee_link.c_str());
  RCLCPP_INFO(logger, "URDF: %s", cfg.urdf_path.c_str());
  RCLCPP_INFO(logger, "SRDF: %s", cfg.srdf_path.c_str());
  RCLCPP_INFO(logger, "Sampling: voxel=%.4f aabb_samples=%d fk_samples=%d fk_max_attempts=%d fk_log_interval=%d",
              cfg.voxel_size, cfg.aabb_samples, cfg.fk_samples, cfg.fk_max_attempts, cfg.fk_log_interval);
  RCLCPP_INFO(logger,
              "Sampling stop: mode=%s interval=%d delta_ratio=%.4f heat_min=%.6f heat_alpha=%.2f patience=%d "
              "min_occupied=%d",
              cfg.coverage_mode.c_str(), cfg.coverage_check_interval, cfg.coverage_delta_ratio, cfg.coverage_heat_min,
              cfg.coverage_heat_alpha, cfg.coverage_patience, cfg.coverage_min_occupied);
  RCLCPP_INFO(logger, "Sampling threads: %d", thread_count);
  RCLCPP_INFO(logger, "Orientation coverage: bins=%d enabled=%s", cfg.orientation_bins,
              cfg.write_orientation_coverage ? "true" : "false");
  RCLCPP_INFO(logger, "IK: attempts=%d timeout=%.4f refine_attempts=%d refine_timeout=%.4f search_disc=%.4f",
              cfg.ik_attempts, cfg.ik_timeout, cfg.ik_attempts_refine, cfg.ik_timeout_refine, cfg.search_discretization);
  RCLCPP_INFO(logger, "SDF: boundary-based EDT (labels 0/1/2)");
  RCLCPP_INFO(logger, "Hole fill: max_voxels=%zu", cfg.hole_fill_max_voxels);
  RCLCPP_INFO(logger, "Closing: radius_voxels=%d", cfg.closing_radius_voxels);
  RCLCPP_INFO(logger, "Output dir: %s", cfg.output_dir.c_str());
  if (cfg.write_hdf5)
  {
    RCLCPP_INFO(logger, "HDF5 output: %s", cfg.hdf5_path.c_str());
  }

  if (!ensureDirectory(cfg.output_dir))
  {
    RCLCPP_ERROR(logger, "Failed to create output dir: %s", cfg.output_dir.c_str());
    rclcpp::shutdown();
    return 1;
  }
  if (cfg.write_hdf5)
  {
    const std::string::size_type slash = cfg.hdf5_path.find_last_of('/');
    if (slash != std::string::npos)
    {
      const std::string hdf5_dir = cfg.hdf5_path.substr(0, slash);
      if (!hdf5_dir.empty() && !ensureDirectory(hdf5_dir))
      {
        RCLCPP_ERROR(logger, "Failed to create hdf5 dir: %s", hdf5_dir.c_str());
        rclcpp::shutdown();
        return 1;
      }
    }
  }

  reachability_cli::RobotContext context;
  try
  {
    context = reachability_cli::loadRobotFromFiles(cfg.urdf_path, cfg.srdf_path);
  }
  catch (const std::exception& ex)
  {
    RCLCPP_ERROR(logger, "%s", ex.what());
    rclcpp::shutdown();
    return 1;
  }

  const auto& robot_model = context.model;
  const auto& scene = context.scene;
  const auto* jmg = robot_model->getJointModelGroup(cfg.group_name);
  if (!jmg)
  {
    RCLCPP_ERROR(logger, "JointModelGroup '%s' not found", cfg.group_name.c_str());
    rclcpp::shutdown();
    return 1;
  }

  if (!robot_model->getLinkModel(cfg.ee_link))
  {
    RCLCPP_ERROR(logger, "Link '%s' not found", cfg.ee_link.c_str());
    rclcpp::shutdown();
    return 1;
  }

  if (!cfg.joint_names.empty() && cfg.joint_names != jmg->getVariableNames())
  {
    RCLCPP_ERROR(logger, "Joint order mismatch for group '%s'", cfg.group_name.c_str());
    rclcpp::shutdown();
    return 1;
  }

  const size_t joint_count = jmg->getVariableCount();
  const std::vector<std::string> joint_names =
      cfg.joint_names.empty() ? jmg->getVariableNames() : cfg.joint_names;

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

  moveit::core::RobotState state(robot_model);
  state.setToDefaultValues();

  collision_detection::CollisionRequest request;
  collision_detection::CollisionResult result;
  request.group_name = cfg.group_name;
  request.contacts = false;
  request.max_contacts = 0;

  int seed_used = cfg.random_seed;
  if (seed_used == 0)
  {
    seed_used = static_cast<int>(std::random_device{}());
  }
  random_numbers::RandomNumberGenerator rng(seed_used);

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

  Eigen::Vector3d aabb_min(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::infinity());
  Eigen::Vector3d aabb_max(-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity());

  size_t aabb_kept = 0;
  const int aabb_report_interval = std::max(1, cfg.aabb_samples / 10);
  for (int i = 0; i < cfg.aabb_samples; ++i)
  {
    state.setToRandomPositions(jmg, rng);
    state.update();
    result.clear();
    scene->checkSelfCollision(request, result, state);
    if (result.collision)
    {
      continue;
    }
    const Eigen::Vector3d pos = state.getGlobalLinkTransform(cfg.ee_link).translation();
    aabb_min = aabb_min.cwiseMin(pos);
    aabb_max = aabb_max.cwiseMax(pos);
    ++aabb_kept;

    if ((i + 1) % aabb_report_interval == 0 || i + 1 == cfg.aabb_samples)
    {
      RCLCPP_INFO(logger, "AABB sampling %d/%d kept=%zu", i + 1, cfg.aabb_samples, aabb_kept);
    }
  }

  if (aabb_kept == 0)
  {
    RCLCPP_ERROR(logger, "Failed to collect any FK samples for AABB");
    rclcpp::shutdown();
    return 1;
  }

  const Eigen::Vector3d aabb_min_raw = aabb_min;
  const Eigen::Vector3d aabb_max_raw = aabb_max;
  const Eigen::Vector3d aabb_center = (aabb_min_raw + aabb_max_raw) * 0.5;
  const Eigen::Vector3d aabb_half_raw = (aabb_max_raw - aabb_min_raw) * 0.5;
  const Eigen::Vector3d aabb_half = aabb_half_raw * cfg.aabb_scale;
  aabb_min = aabb_center - aabb_half;
  aabb_max = aabb_center + aabb_half;

  Grid grid;
  grid.origin = aabb_min;
  grid.voxel_size = cfg.voxel_size;
  grid.nx = static_cast<size_t>(std::ceil((aabb_max.x() - aabb_min.x()) / cfg.voxel_size));
  grid.ny = static_cast<size_t>(std::ceil((aabb_max.y() - aabb_min.y()) / cfg.voxel_size));
  grid.nz = static_cast<size_t>(std::ceil((aabb_max.z() - aabb_min.z()) / cfg.voxel_size));

  if (grid.nx == 0 || grid.ny == 0 || grid.nz == 0)
  {
    RCLCPP_ERROR(logger, "Invalid grid dimensions");
    rclcpp::shutdown();
    return 1;
  }

  const size_t grid_size = grid.size();
  RCLCPP_INFO(logger, "Grid dims: %zu x %zu x %zu (%.3f mm)", grid.nx, grid.ny, grid.nz, cfg.voxel_size * 1000.0);

  std::vector<ThreadContext> thread_contexts;
  thread_contexts.reserve(static_cast<size_t>(thread_count));
  for (int i = 0; i < thread_count; ++i)
  {
    const int thread_seed = seed_used + 7919 * (i + 1);
    thread_contexts.emplace_back(robot_model, thread_seed, joint_count);
  }

  Hdf5Writer hdf5_writer;
  if (cfg.write_hdf5)
  {
    if (!hdf5_writer.open(cfg.hdf5_path, cfg.hdf5_chunk, joint_count))
    {
      RCLCPP_ERROR(logger, "Failed to create HDF5 output: %s", cfg.hdf5_path.c_str());
      rclcpp::shutdown();
      return 1;
    }
    hdf5_writer.writeString(hdf5_writer.meta_group, "robot_name", cfg.robot_name);
    hdf5_writer.writeString(hdf5_writer.meta_group, "group_name", cfg.group_name);
    hdf5_writer.writeString(hdf5_writer.meta_group, "base_link", cfg.base_link);
    hdf5_writer.writeString(hdf5_writer.meta_group, "ee_link", cfg.ee_link);
    hdf5_writer.writeStringArray(hdf5_writer.meta_group, "joint_names", joint_names);
    hdf5_writer.writeString(hdf5_writer.meta_group, "config_path", options.config_path);
    if (!config_yaml.empty())
    {
      hdf5_writer.writeString(hdf5_writer.meta_group, "config_yaml", config_yaml);
    }
  }

  const bool track_orientation = cfg.write_orientation_coverage && cfg.orientation_bins > 0;
  const size_t orientation_words = track_orientation ? static_cast<size_t>((cfg.orientation_bins + 63) / 64) : 0;
  std::vector<std::array<float, 4>> orientation_bins;
  std::vector<uint64_t> orientation_bits;
  if (track_orientation)
  {
    orientation_bins.reserve(static_cast<size_t>(cfg.orientation_bins));
    random_numbers::RandomNumberGenerator orientation_rng(seed_used + 131);
    for (int i = 0; i < cfg.orientation_bins; ++i)
    {
      const Eigen::Quaterniond q = sampleUniformQuaternion(orientation_rng);
      orientation_bins.push_back(
          { static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z()),
            static_cast<float>(q.w()) });
    }

    const size_t bit_count = grid_size * orientation_words;
    orientation_bits.assign(bit_count, 0);
    const double bytes = static_cast<double>(bit_count) * sizeof(uint64_t);
    RCLCPP_INFO(logger, "Orientation coverage buffer: %.2f MB", bytes / (1024.0 * 1024.0));
  }

  std::vector<std::atomic<uint32_t>> voxel_counts(grid_size);
  for (size_t i = 0; i < grid_size; ++i)
  {
    voxel_counts[i].store(0, std::memory_order_relaxed);
  }
  std::atomic<size_t> occupied_voxels_atomic{ 0 };

  const bool write_samples_hdf5 = cfg.write_hdf5 && cfg.write_fk_samples;
  const bool write_samples_npy = cfg.write_npy && cfg.write_fk_samples;
  std::vector<SampleBuffer> sample_buffers;
  if (write_samples_hdf5)
  {
    sample_buffers.resize(static_cast<size_t>(thread_count));
  }

  const bool keep_fk_samples = write_samples_npy;
  std::vector<float> fk_positions;
  std::vector<float> fk_quats;
  const size_t fk_capacity = keep_fk_samples ? static_cast<size_t>(cfg.fk_samples) : 0;
  if (keep_fk_samples && fk_capacity > 0)
  {
    fk_positions.resize(fk_capacity * 3);
    fk_quats.resize(fk_capacity * 4);
  }

  std::atomic<size_t> fk_kept_atomic{ 0 };
  std::atomic<size_t> fk_store_atomic{ 0 };
  size_t fk_attempts = 0;
  size_t fk_kept = 0;
  size_t fk_stored = 0;
  const size_t fk_report_interval =
      std::max<size_t>(1, static_cast<size_t>(cfg.fk_log_interval > 0 ? cfg.fk_log_interval : cfg.fk_max_attempts / 20));
  size_t next_log_at = fk_report_interval;
  const int coverage_interval = std::max(1, cfg.coverage_check_interval);
  size_t last_occupied_voxels = 0;
  size_t last_attempts = 0;
  int stagnation_count = 0;
  double coverage_heat_ema = -1.0;
  while (fk_attempts < static_cast<size_t>(cfg.fk_max_attempts) &&
         fk_kept < static_cast<size_t>(cfg.fk_samples))
  {
    const size_t remaining_attempts = static_cast<size_t>(cfg.fk_max_attempts) - fk_attempts;
    const size_t batch_attempts = std::min(remaining_attempts, static_cast<size_t>(coverage_interval));

    if (write_samples_hdf5)
    {
      const size_t reserve_count = std::max<size_t>(1, batch_attempts / static_cast<size_t>(thread_count));
      for (auto& buffer : sample_buffers)
      {
        buffer.clear();
        buffer.reserve(reserve_count, joint_count);
      }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t attempt_idx = 0; attempt_idx < batch_attempts; ++attempt_idx)
    {
      int tid = 0;
#ifdef _OPENMP
      tid = omp_get_thread_num();
#endif
      auto& ctx = thread_contexts[static_cast<size_t>(tid)];
      ctx.state.setToRandomPositions(jmg, ctx.rng);
      ctx.state.update();
      ctx.result.clear();
      scene->checkSelfCollision(request, ctx.result, ctx.state);
      if (ctx.result.collision)
      {
        continue;
      }

      const Eigen::Isometry3d& tf = ctx.state.getGlobalLinkTransform(cfg.ee_link);
      const Eigen::Vector3d pos = tf.translation();
      Eigen::Quaterniond quat(tf.rotation());
      quat.normalize();

      const double gx = (pos.x() - grid.origin.x()) / grid.voxel_size;
      const double gy = (pos.y() - grid.origin.y()) / grid.voxel_size;
      const double gz = (pos.z() - grid.origin.z()) / grid.voxel_size;
      if (gx < 0.0 || gy < 0.0 || gz < 0.0 || gx >= static_cast<double>(grid.nx) ||
          gy >= static_cast<double>(grid.ny) || gz >= static_cast<double>(grid.nz))
      {
        continue;
      }

      const size_t xi = static_cast<size_t>(gx);
      const size_t yi = static_cast<size_t>(gy);
      const size_t zi = static_cast<size_t>(gz);
      const size_t idx = grid.index(xi, yi, zi);

      if (track_orientation)
      {
        const float qx = static_cast<float>(quat.x());
        const float qy = static_cast<float>(quat.y());
        const float qz = static_cast<float>(quat.z());
        const float qw = static_cast<float>(quat.w());
        float best = -1.0f;
        int best_idx = 0;
        for (int i = 0; i < cfg.orientation_bins; ++i)
        {
          const auto& b = orientation_bins[static_cast<size_t>(i)];
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
        const size_t word_base = idx * orientation_words;
        const size_t word_index = word_base + static_cast<size_t>(best_idx / 64);
        const uint64_t mask = 1ULL << (best_idx % 64);
        __atomic_fetch_or(&orientation_bits[word_index], mask, __ATOMIC_RELAXED);
      }

      const uint32_t prev =
          voxel_counts[idx].fetch_add(1, std::memory_order_relaxed);
      if (prev == 0)
      {
        occupied_voxels_atomic.fetch_add(1, std::memory_order_relaxed);
      }

      fk_kept_atomic.fetch_add(1, std::memory_order_relaxed);

      if (write_samples_hdf5)
      {
        SampleBuffer& buffer = sample_buffers[static_cast<size_t>(tid)];
        buffer.positions.push_back(static_cast<float>(pos.x()));
        buffer.positions.push_back(static_cast<float>(pos.y()));
        buffer.positions.push_back(static_cast<float>(pos.z()));
        buffer.quats.push_back(static_cast<float>(quat.x()));
        buffer.quats.push_back(static_cast<float>(quat.y()));
        buffer.quats.push_back(static_cast<float>(quat.z()));
        buffer.quats.push_back(static_cast<float>(quat.w()));
        ctx.state.copyJointGroupPositions(jmg, ctx.joint_positions);
        for (size_t j = 0; j < ctx.joint_positions.size(); ++j)
        {
          buffer.joints.push_back(static_cast<float>(ctx.joint_positions[j]));
        }
        buffer.voxels.push_back(static_cast<uint64_t>(idx));
      }

      if (!keep_fk_samples || fk_capacity == 0)
      {
        continue;
      }

      const size_t store_idx = fk_store_atomic.fetch_add(1, std::memory_order_relaxed);
      if (store_idx >= fk_capacity)
      {
        continue;
      }

      const size_t pos_offset = store_idx * 3;
      fk_positions[pos_offset] = static_cast<float>(pos.x());
      fk_positions[pos_offset + 1] = static_cast<float>(pos.y());
      fk_positions[pos_offset + 2] = static_cast<float>(pos.z());

      const size_t quat_offset = store_idx * 4;
      fk_quats[quat_offset] = static_cast<float>(quat.x());
      fk_quats[quat_offset + 1] = static_cast<float>(quat.y());
      fk_quats[quat_offset + 2] = static_cast<float>(quat.z());
      fk_quats[quat_offset + 3] = static_cast<float>(quat.w());
    }

    if (write_samples_hdf5)
    {
      for (auto& buffer : sample_buffers)
      {
        if (buffer.size() == 0)
        {
          continue;
        }
        if (!hdf5_writer.appendSamples(buffer))
        {
          RCLCPP_ERROR(logger, "Failed to append HDF5 samples.");
          hdf5_writer.close();
          rclcpp::shutdown();
          return 1;
        }
      }
    }

    fk_attempts += batch_attempts;
    fk_kept = fk_kept_atomic.load(std::memory_order_relaxed);
    if (write_samples_hdf5)
    {
      fk_stored = hdf5_writer.sample_count;
    }
    else
    {
      fk_stored = keep_fk_samples ? std::min(fk_store_atomic.load(std::memory_order_relaxed), fk_capacity) : 0;
    }
    const size_t occupied_voxels = occupied_voxels_atomic.load(std::memory_order_relaxed);

    if (fk_attempts >= next_log_at || fk_attempts >= static_cast<size_t>(cfg.fk_max_attempts) ||
        fk_kept >= static_cast<size_t>(cfg.fk_samples))
    {
      const double accept_rate = fk_attempts > 0 ? static_cast<double>(fk_kept) / static_cast<double>(fk_attempts) : 0.0;
      const double coverage_pct =
          grid_size > 0 ? (static_cast<double>(occupied_voxels) / static_cast<double>(grid_size)) * 100.0 : 0.0;
      RCLCPP_INFO(logger,
                  "FK sampling attempts=%zu/%d kept=%zu/%d accept=%.2f%% occupied_voxels=%zu coverage=%.2f%%",
                  fk_attempts, cfg.fk_max_attempts, fk_kept, cfg.fk_samples, accept_rate * 100.0, occupied_voxels,
                  coverage_pct);
      next_log_at = fk_attempts + fk_report_interval;
    }

    const size_t new_occupied = occupied_voxels - last_occupied_voxels;
    const size_t attempts_window = fk_attempts - last_attempts;
    bool below = false;
    const double coverage_pct =
        grid_size > 0 ? (static_cast<double>(occupied_voxels) / static_cast<double>(grid_size)) * 100.0 : 0.0;
    const double delta_pct =
        grid_size > 0 ? (static_cast<double>(new_occupied) / static_cast<double>(grid_size)) * 100.0 : 0.0;
    if (cfg.coverage_mode == "heat")
    {
      const double heat =
          attempts_window > 0 ? static_cast<double>(new_occupied) / static_cast<double>(attempts_window) : 0.0;
      if (coverage_heat_ema < 0.0)
      {
        coverage_heat_ema = heat;
      }
      else
      {
        coverage_heat_ema = cfg.coverage_heat_alpha * heat + (1.0 - cfg.coverage_heat_alpha) * coverage_heat_ema;
      }
      if (occupied_voxels >= static_cast<size_t>(cfg.coverage_min_occupied) && coverage_heat_ema < cfg.coverage_heat_min)
      {
        below = true;
      }
      if (below)
      {
        ++stagnation_count;
      }
      else
      {
        stagnation_count = 0;
      }
      const double heat_pct = heat * 100.0;
      const double heat_ema_pct = coverage_heat_ema * 100.0;
      const double heat_threshold_pct = cfg.coverage_heat_min * 100.0;
      RCLCPP_INFO(logger,
                  "Coverage heat: new=%zu (%.5f%%) rate=%.5f%% ema=%.5f%% threshold=%.5f%% stagnation=%d/%d "
                  "occupied=%zu (%.2f%%)",
                  new_occupied, delta_pct, heat_pct, heat_ema_pct, heat_threshold_pct, stagnation_count,
                  cfg.coverage_patience, occupied_voxels, coverage_pct);
    }
    else
    {
      const double ratio =
          last_occupied_voxels > 0 ? static_cast<double>(new_occupied) / static_cast<double>(last_occupied_voxels) : 1.0;
      if (occupied_voxels >= static_cast<size_t>(cfg.coverage_min_occupied) && ratio < cfg.coverage_delta_ratio)
      {
        below = true;
      }
      if (below)
      {
        ++stagnation_count;
      }
      else
      {
        stagnation_count = 0;
      }
      RCLCPP_INFO(logger,
                  "Coverage delta: new=%zu (%.5f%%) ratio=%.5f stagnation=%d/%d occupied=%zu (%.2f%%)",
                  new_occupied, delta_pct, ratio, stagnation_count, cfg.coverage_patience, occupied_voxels,
                  coverage_pct);
    }

    last_occupied_voxels = occupied_voxels;
    last_attempts = fk_attempts;

    if (stagnation_count >= cfg.coverage_patience)
    {
      RCLCPP_INFO(logger, "Coverage converged. Stop FK sampling early.");
      break;
    }
  }

  if (fk_kept == 0)
  {
    RCLCPP_ERROR(logger, "Failed to collect FK samples for labels");
    rclcpp::shutdown();
    return 1;
  }

  std::vector<uint8_t> labels(grid_size, static_cast<uint8_t>(CellLabel::kOutside));
  std::vector<float> sdf(grid_size, std::numeric_limits<float>::quiet_NaN());
  size_t inside_count = 0;
  size_t outside_count = grid_size;
  size_t boundary_count = 0;

  for (size_t i = 0; i < grid_size; ++i)
  {
    if (voxel_counts[i].load(std::memory_order_relaxed) > 0)
    {
      labels[i] = static_cast<uint8_t>(CellLabel::kInside);
      ++inside_count;
      if (outside_count > 0)
      {
        --outside_count;
      }
    }
  }

  if (cfg.hole_fill_max_voxels > 0)
  {
    const size_t filled = fillSmallHoles(labels, grid, cfg.hole_fill_max_voxels, logger);
    inside_count += filled;
    if (outside_count >= filled)
    {
      outside_count -= filled;
    }
    else
    {
      outside_count = 0;
    }
  }

  if (cfg.closing_radius_voxels > 0)
  {
    inside_count = applyClosing(labels, grid, cfg.closing_radius_voxels, logger);
    outside_count = grid_size - inside_count;
  }

  const int neighbor_offsets[6][3] = {
    {1, 0, 0},  {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
  };
  for (size_t x = 0; x < grid.nx; ++x)
  {
    for (size_t y = 0; y < grid.ny; ++y)
    {
      for (size_t z_idx = 0; z_idx < grid.nz; ++z_idx)
      {
        const size_t idx = grid.index(x, y, z_idx);
        if (labels[idx] != static_cast<uint8_t>(CellLabel::kInside))
        {
          continue;
        }
        bool boundary = false;
        for (const auto& off : neighbor_offsets)
        {
          const int nx = static_cast<int>(x) + off[0];
          const int ny = static_cast<int>(y) + off[1];
          const int nz = static_cast<int>(z_idx) + off[2];
          if (nx < 0 || ny < 0 || nz < 0 || nx >= static_cast<int>(grid.nx) ||
              ny >= static_cast<int>(grid.ny) || nz >= static_cast<int>(grid.nz))
          {
            continue;
          }
          const size_t nidx =
              grid.index(static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz));
          if (labels[nidx] == static_cast<uint8_t>(CellLabel::kOutside))
          {
            boundary = true;
            break;
          }
        }
        if (boundary)
        {
          labels[idx] = static_cast<uint8_t>(CellLabel::kBoundary);
          ++boundary_count;
        }
      }
    }
  }
  if (inside_count >= boundary_count)
  {
    inside_count -= boundary_count;
  }
  else
  {
    inside_count = 0;
  }

  if (cfg.write_sdf_grid)
  {
    if (boundary_count == 0)
    {
      RCLCPP_WARN(logger, "No boundary voxels found; SDF will remain NaN.");
    }
    else
    {
      std::vector<float> dist_to_boundary(grid_size, 1e20f);
      for (size_t i = 0; i < grid_size; ++i)
      {
        if (labels[i] == static_cast<uint8_t>(CellLabel::kBoundary))
        {
          dist_to_boundary[i] = 0.0f;
        }
      }
      RCLCPP_INFO(logger, "EDT: distance to boundary");
      distanceTransform3D(dist_to_boundary, grid);
      for (size_t i = 0; i < grid_size; ++i)
      {
        const uint8_t label = labels[i];
        if (label == static_cast<uint8_t>(CellLabel::kBoundary))
        {
          sdf[i] = 0.0f;
        }
        else if (label == static_cast<uint8_t>(CellLabel::kInside))
        {
          sdf[i] = std::sqrt(dist_to_boundary[i]) * static_cast<float>(grid.voxel_size);
        }
        else if (label == static_cast<uint8_t>(CellLabel::kOutside))
        {
          sdf[i] = -std::sqrt(dist_to_boundary[i]) * static_cast<float>(grid.voxel_size);
        }
      }
    }
  }

  std::vector<float> orientation_coverage;
  std::vector<float> orientation_bins_out;
  if (track_orientation)
  {
    orientation_coverage.resize(grid_size, 0.0f);
    for (size_t idx = 0; idx < grid_size; ++idx)
    {
      size_t count = 0;
      const size_t word_base = idx * orientation_words;
      for (size_t w = 0; w < orientation_words; ++w)
      {
        const uint64_t bits = orientation_bits[word_base + w];
        count += static_cast<size_t>(__builtin_popcountll(bits));
      }
      orientation_coverage[idx] = static_cast<float>(count) / static_cast<float>(cfg.orientation_bins);
    }

    orientation_bins_out.reserve(static_cast<size_t>(cfg.orientation_bins) * 4);
    for (const auto& b : orientation_bins)
    {
      orientation_bins_out.push_back(b[0]);
      orientation_bins_out.push_back(b[1]);
      orientation_bins_out.push_back(b[2]);
      orientation_bins_out.push_back(b[3]);
    }
  }

  std::vector<uint32_t> voxel_counts_out;
  if (cfg.write_voxel_counts || cfg.write_hdf5)
  {
    voxel_counts_out.resize(grid_size, 0);
    for (size_t i = 0; i < grid_size; ++i)
    {
      voxel_counts_out[i] = voxel_counts[i].load(std::memory_order_relaxed);
    }
  }

  if (cfg.write_hdf5)
  {
    const std::vector<hsize_t> grid_dims{ grid.nx, grid.ny, grid.nz };
    const std::vector<double> origin_vec = { grid.origin.x(), grid.origin.y(), grid.origin.z() };
    const std::vector<uint64_t> dims_vec = { grid.nx, grid.ny, grid.nz };
    const double voxel_val = grid.voxel_size;
    hdf5_writer.writeArray(hdf5_writer.grid_group, "origin", H5T_IEEE_F64LE, { 3 }, origin_vec.data());
    hdf5_writer.writeArray(hdf5_writer.grid_group, "dims", H5T_STD_U64LE, { 3 }, dims_vec.data());
    hdf5_writer.writeArray(hdf5_writer.grid_group, "voxel_size", H5T_IEEE_F64LE, { 1 }, &voxel_val);
    hdf5_writer.writeArray(hdf5_writer.grid_group, "label", H5T_STD_U8LE, grid_dims, labels.data());
    if (!voxel_counts_out.empty())
    {
      hdf5_writer.writeArray(hdf5_writer.grid_group, "voxel_counts", H5T_STD_U32LE, grid_dims,
                             voxel_counts_out.data());
    }
    if (cfg.write_sdf_grid)
    {
      hdf5_writer.writeArray(hdf5_writer.grid_group, "sdf", H5T_IEEE_F32LE, grid_dims, sdf.data());
    }
    if (!orientation_coverage.empty())
    {
      hdf5_writer.writeArray(hdf5_writer.grid_group, "orientation_coverage", H5T_IEEE_F32LE, grid_dims,
                             orientation_coverage.data());
    }
    if (!orientation_bins_out.empty())
    {
      const size_t bin_count = orientation_bins_out.size() / 4;
      hdf5_writer.writeArray(hdf5_writer.grid_group, "orientation_bins", H5T_IEEE_F32LE,
                             { static_cast<hsize_t>(bin_count), 4 }, orientation_bins_out.data());
    }
    const std::string metadata_yaml =
        buildMetadataYaml(cfg, grid, aabb_min, aabb_max, fk_kept, fk_attempts, fk_stored, inside_count, outside_count,
                          boundary_count, seed_used);
    hdf5_writer.writeString(hdf5_writer.meta_group, "metadata_yaml", metadata_yaml);

    if (write_samples_hdf5)
    {
      if (!writeCsrToHdf5(hdf5_writer, grid, voxel_counts_out, logger))
      {
        hdf5_writer.close();
        rclcpp::shutdown();
        return 1;
      }
    }
  }

  if (cfg.write_npy)
  {
    const std::string sdf_path = cfg.output_dir + "/sdf.npy";
    const std::string label_path = cfg.output_dir + "/label.npy";
    const std::string voxel_counts_path = cfg.output_dir + "/voxel_counts.npy";
    const std::string orientation_cov_path = cfg.output_dir + "/orientation_coverage.npy";
    const std::string orientation_bins_path = cfg.output_dir + "/orientation_bins.npy";
    const std::string origin_path = cfg.output_dir + "/origin.npy";
    const std::string dims_path = cfg.output_dir + "/dims.npy";
    const std::string voxel_size_path = cfg.output_dir + "/voxel_size.npy";
    const std::string fk_pos_path = cfg.output_dir + "/fk_positions.npy";
    const std::string fk_quat_path = cfg.output_dir + "/fk_quats.npy";

    std::vector<size_t> grid_shape{ grid.nx, grid.ny, grid.nz };
    if (cfg.write_sdf_grid)
    {
      RCLCPP_INFO(logger, "Writing %s", sdf_path.c_str());
      reachability_cli::writeNpy(sdf_path, sdf.data(), grid_shape, reachability_cli::NpyDataType::kFloat32);
    }
    RCLCPP_INFO(logger, "Writing %s", label_path.c_str());
    reachability_cli::writeNpy(label_path, labels.data(), grid_shape, reachability_cli::NpyDataType::kUInt8);
    if (cfg.write_voxel_counts && !voxel_counts_out.empty())
    {
      RCLCPP_INFO(logger, "Writing %s", voxel_counts_path.c_str());
      reachability_cli::writeNpy(voxel_counts_path, voxel_counts_out.data(), grid_shape,
                                 reachability_cli::NpyDataType::kUInt32);
    }

    if (!orientation_coverage.empty())
    {
      RCLCPP_INFO(logger, "Writing %s", orientation_cov_path.c_str());
      reachability_cli::writeNpy(orientation_cov_path, orientation_coverage.data(), grid_shape,
                                 reachability_cli::NpyDataType::kFloat32);
      if (!orientation_bins_out.empty())
      {
        const size_t bin_count = orientation_bins_out.size() / 4;
        std::vector<size_t> bin_shape{ bin_count, 4 };
        RCLCPP_INFO(logger, "Writing %s", orientation_bins_path.c_str());
        reachability_cli::writeNpy(orientation_bins_path, orientation_bins_out.data(), bin_shape,
                                   reachability_cli::NpyDataType::kFloat32);
      }
    }

    if (cfg.write_fk_samples && fk_stored > 0 && keep_fk_samples)
    {
      const size_t fk_count = fk_stored;
      std::vector<size_t> fk_pos_shape{ fk_count, 3 };
      std::vector<size_t> fk_quat_shape{ fk_count, 4 };
      reachability_cli::writeNpy(fk_pos_path, fk_positions.data(), fk_pos_shape, reachability_cli::NpyDataType::kFloat32);
      reachability_cli::writeNpy(fk_quat_path, fk_quats.data(), fk_quat_shape, reachability_cli::NpyDataType::kFloat32);
    }

    const std::vector<float> origin_vec = { static_cast<float>(grid.origin.x()), static_cast<float>(grid.origin.y()),
                                            static_cast<float>(grid.origin.z()) };
    const std::vector<int32_t> dims_vec = { static_cast<int32_t>(grid.nx), static_cast<int32_t>(grid.ny),
                                            static_cast<int32_t>(grid.nz) };
    const std::vector<float> voxel_vec = { static_cast<float>(grid.voxel_size) };
    RCLCPP_INFO(logger, "Writing %s", origin_path.c_str());
    reachability_cli::writeNpy(origin_path, origin_vec.data(), { 3 }, reachability_cli::NpyDataType::kFloat32);
    RCLCPP_INFO(logger, "Writing %s", dims_path.c_str());
    reachability_cli::writeNpy(dims_path, dims_vec.data(), { 3 }, reachability_cli::NpyDataType::kInt32);
    RCLCPP_INFO(logger, "Writing %s", voxel_size_path.c_str());
    reachability_cli::writeNpy(voxel_size_path, voxel_vec.data(), { 1 }, reachability_cli::NpyDataType::kFloat32);
  }

  if (cfg.write_metadata_yaml)
  {
    RCLCPP_INFO(logger, "Writing %s", (cfg.output_dir + "/metadata.yaml").c_str());
    writeMetadata(cfg.output_dir + "/metadata.yaml", cfg, grid, aabb_min, aabb_max, fk_kept, fk_attempts, fk_stored,
                  inside_count, outside_count, boundary_count, seed_used);
  }

  if (cfg.write_hdf5)
  {
    hdf5_writer.close();
  }

  RCLCPP_INFO(logger, "Done. Output: %s", cfg.output_dir.c_str());
  rclcpp::shutdown();
  return 0;
}
