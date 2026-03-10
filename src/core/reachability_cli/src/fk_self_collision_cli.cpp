// File: fk_self_collision_cli.cpp
// Purpose: Evaluate FK poses and self-collision status for sampled joint configurations.
// Usage: ros2 run reachability_cli fk_self_collision_cli --urdf <urdf> --srdf <srdf> ...

#include "reachability_cli/model_loader.h"

#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <moveit/collision_detection/collision_common.h>
#include <moveit/robot_state/robot_state.h>

namespace
{
struct Options
{
  std::string urdf_path;
  std::string srdf_path;
  std::string group_name;
  std::string ee_link;
  std::string joint_names_csv;
  std::string joint_values_csv;
  std::string input_path;
  std::string output_path;
};

void printUsage()
{
  std::cerr
      << "Usage: fk_self_collision_cli --urdf <urdf> --srdf <srdf> --group <name> --ee-link <link> \\\n"
         "       [--joint-names <csv>] [--joint-values <csv> | --input <file>] [--output <file>]\n\n"
         "Input format (file): one sample per line, values separated by space or comma.\n"
         "Output columns: within_limits self_collision_free reachable x y z qx qy qz qw\n";
}

std::vector<std::string> parseStringList(const std::string& input)
{
  std::string normalized = input;
  std::replace(normalized.begin(), normalized.end(), ',', ' ');
  std::istringstream iss(normalized);
  std::vector<std::string> tokens;
  std::string token;
  while (iss >> token)
  {
    tokens.push_back(token);
  }
  return tokens;
}

std::vector<double> parseDoubleList(const std::string& input)
{
  std::string normalized = input;
  std::replace(normalized.begin(), normalized.end(), ',', ' ');
  std::istringstream iss(normalized);
  std::vector<double> values;
  double value = 0.0;
  while (iss >> value)
  {
    values.push_back(value);
  }
  return values;
}

bool parseArgs(int argc, char** argv, Options& options)
{
  for (int i = 1; i < argc; ++i)
  {
    std::string arg = argv[i];
    if (arg == "--urdf" && i + 1 < argc)
    {
      options.urdf_path = argv[++i];
    }
    else if (arg == "--srdf" && i + 1 < argc)
    {
      options.srdf_path = argv[++i];
    }
    else if (arg == "--group" && i + 1 < argc)
    {
      options.group_name = argv[++i];
    }
    else if (arg == "--ee-link" && i + 1 < argc)
    {
      options.ee_link = argv[++i];
    }
    else if (arg == "--joint-names" && i + 1 < argc)
    {
      options.joint_names_csv = argv[++i];
    }
    else if (arg == "--joint-values" && i + 1 < argc)
    {
      options.joint_values_csv = argv[++i];
    }
    else if (arg == "--input" && i + 1 < argc)
    {
      options.input_path = argv[++i];
    }
    else if (arg == "--output" && i + 1 < argc)
    {
      options.output_path = argv[++i];
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

  if (options.urdf_path.empty() || options.srdf_path.empty() || options.group_name.empty() ||
      options.ee_link.empty())
  {
    return false;
  }

  if (options.joint_values_csv.empty() && options.input_path.empty())
  {
    return false;
  }

  if (!options.joint_values_csv.empty() && !options.input_path.empty())
  {
    std::cerr << "Provide either --joint-values or --input, not both.\n";
    return false;
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

  rclcpp::Logger logger = rclcpp::get_logger("fk_self_collision_cli");

  reachability_cli::RobotContext context;
  try
  {
    context = reachability_cli::loadRobotFromFiles(options.urdf_path, options.srdf_path);
  }
  catch (const std::exception& ex)
  {
    RCLCPP_ERROR(logger, "%s", ex.what());
    rclcpp::shutdown();
    return 1;
  }

  const auto& robot_model = context.model;
  const auto& scene = context.scene;

  const auto* jmg = robot_model->getJointModelGroup(options.group_name);
  if (!jmg)
  {
    RCLCPP_ERROR(logger, "JointModelGroup '%s' not found", options.group_name.c_str());
    rclcpp::shutdown();
    return 1;
  }

  if (!robot_model->getLinkModel(options.ee_link))
  {
    RCLCPP_ERROR(logger, "Link '%s' not found", options.ee_link.c_str());
    rclcpp::shutdown();
    return 1;
  }

  const auto& variable_names = jmg->getVariableNames();
  const std::vector<std::string> declared_names =
      options.joint_names_csv.empty() ? std::vector<std::string>() : parseStringList(options.joint_names_csv);
  if (!declared_names.empty() && declared_names != variable_names)
  {
    RCLCPP_ERROR(logger, "Provided joint order does not match group variable order");
    rclcpp::shutdown();
    return 1;
  }

  moveit::core::RobotState state(robot_model);
  state.setToDefaultValues();

  collision_detection::CollisionRequest request;
  collision_detection::CollisionResult result;
  request.group_name = options.group_name;
  request.contacts = false;
  request.max_contacts = 0;

  std::ofstream output_file;
  std::ostream* output_stream = &std::cout;
  if (!options.output_path.empty())
  {
    output_file.open(options.output_path);
    if (!output_file)
    {
      RCLCPP_ERROR(logger, "Failed to open output file: %s", options.output_path.c_str());
      rclcpp::shutdown();
      return 1;
    }
    output_stream = &output_file;
  }

  auto emit_result = [&](const std::vector<double>& values) -> bool {
    if (values.size() != variable_names.size())
    {
      RCLCPP_ERROR(logger, "Joint vector size mismatch: expected %zu, got %zu", variable_names.size(), values.size());
      return false;
    }

    state.setJointGroupPositions(jmg, values);
    state.update();

    const bool within_limits = state.satisfiesBounds(jmg);

    result.clear();
    scene->checkSelfCollision(request, result, state);
    const bool self_collision_free = !result.collision;

    const Eigen::Isometry3d& tf = state.getGlobalLinkTransform(options.ee_link);
    const Eigen::Quaterniond quat(tf.rotation());

    (*output_stream) << (within_limits ? 1 : 0) << ' ' << (self_collision_free ? 1 : 0) << ' '
                     << ((within_limits && self_collision_free) ? 1 : 0) << ' ' << tf.translation().x() << ' '
                     << tf.translation().y() << ' ' << tf.translation().z() << ' ' << quat.x() << ' ' << quat.y() << ' '
                     << quat.z() << ' ' << quat.w() << '\n';

    return true;
  };

  if (!options.input_path.empty())
  {
    std::ifstream input_file(options.input_path);
    if (!input_file)
    {
      RCLCPP_ERROR(logger, "Failed to open input file: %s", options.input_path.c_str());
      rclcpp::shutdown();
      return 1;
    }

    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input_file, line))
    {
      ++line_number;
      if (line.empty())
      {
        continue;
      }
      const std::vector<double> values = parseDoubleList(line);
      if (!emit_result(values))
      {
        RCLCPP_ERROR(logger, "Failed on line %zu", line_number);
        rclcpp::shutdown();
        return 1;
      }
    }
  }
  else
  {
    const std::vector<double> values = parseDoubleList(options.joint_values_csv);
    if (!emit_result(values))
    {
      rclcpp::shutdown();
      return 1;
    }
  }

  rclcpp::shutdown();
  return 0;
}
