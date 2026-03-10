// File: ik_self_collision_cli.cpp
// Purpose: Solve IK requests and report self-collision validity for target poses.
// Usage: ros2 run reachability_cli ik_self_collision_cli --urdf <urdf> --srdf <srdf> ...

#include "reachability_cli/model_loader.h"

#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>

#include <moveit/collision_detection/collision_common.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_model/joint_model_group.h>
#include <moveit/kdl_kinematics_plugin/kdl_kinematics_plugin.h>

namespace
{
struct Options
{
  std::string urdf_path;
  std::string srdf_path;
  std::string group_name;
  std::string base_link;
  std::string ee_link;
  std::string seed_csv;
  std::string pose_csv;
  std::string input_path;
  std::string output_path;
  double ik_timeout = 0.005;
  double search_discretization = 0.005;
  int ik_attempts = 1;
};

void printUsage()
{
  std::cerr
      << "Usage: ik_self_collision_cli --urdf <urdf> --srdf <srdf> --group <name> \\\n"
         "       --base-link <link> --ee-link <link> [--seed <csv>] [--pose <csv> | --input <file>] \\\n"
         "       [--ik-timeout <sec>] [--ik-attempts <n>] [--search-discretization <value>] [--output <file>]\n\n"
         "Input format (file): one pose per line as x y z qx qy qz qw (space or comma separated).\n"
         "Output columns: ik_success self_collision_free reachable q1 q2 q3 q4 q5 q6\n";
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
    else if (arg == "--base-link" && i + 1 < argc)
    {
      options.base_link = argv[++i];
    }
    else if (arg == "--ee-link" && i + 1 < argc)
    {
      options.ee_link = argv[++i];
    }
    else if (arg == "--seed" && i + 1 < argc)
    {
      options.seed_csv = argv[++i];
    }
    else if (arg == "--pose" && i + 1 < argc)
    {
      options.pose_csv = argv[++i];
    }
    else if (arg == "--input" && i + 1 < argc)
    {
      options.input_path = argv[++i];
    }
    else if (arg == "--output" && i + 1 < argc)
    {
      options.output_path = argv[++i];
    }
    else if (arg == "--ik-timeout" && i + 1 < argc)
    {
      options.ik_timeout = std::stod(argv[++i]);
    }
    else if (arg == "--ik-attempts" && i + 1 < argc)
    {
      options.ik_attempts = std::stoi(argv[++i]);
    }
    else if (arg == "--search-discretization" && i + 1 < argc)
    {
      options.search_discretization = std::stod(argv[++i]);
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
      options.base_link.empty() || options.ee_link.empty())
  {
    return false;
  }

  if (options.pose_csv.empty() && options.input_path.empty())
  {
    return false;
  }

  if (!options.pose_csv.empty() && !options.input_path.empty())
  {
    std::cerr << "Provide either --pose or --input, not both.\n";
    return false;
  }

  if (options.ik_attempts < 1)
  {
    std::cerr << "--ik-attempts must be >= 1\n";
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

  auto node = rclcpp::Node::make_shared("ik_self_collision_cli");
  rclcpp::Logger logger = node->get_logger();

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

  auto solver_allocator = [node, robot_model, options](const moveit::core::JointModelGroup* group)
      -> kinematics::KinematicsBasePtr {
    auto solver = std::make_shared<kdl_kinematics_plugin::KDLKinematicsPlugin>();
    const std::vector<std::string> tips{ options.ee_link };
    if (!solver->initialize(node, *robot_model, group->getName(), options.base_link, tips, options.search_discretization))
    {
      RCLCPP_ERROR(node->get_logger(), "Failed to initialize KDL solver for group '%s'", group->getName().c_str());
      return kinematics::KinematicsBasePtr();
    }
    return kinematics::KinematicsBasePtr(solver);
  };

  const_cast<moveit::core::JointModelGroup*>(jmg)->setSolverAllocators(solver_allocator);
  const_cast<moveit::core::JointModelGroup*>(jmg)->setDefaultIKTimeout(options.ik_timeout);

  moveit::core::RobotState state(robot_model);
  state.setToDefaultValues();

  if (!options.seed_csv.empty())
  {
    const std::vector<double> seed = parseDoubleList(options.seed_csv);
    if (seed.size() != jmg->getVariableCount())
    {
      RCLCPP_ERROR(logger, "Seed size mismatch: expected %u", jmg->getVariableCount());
      rclcpp::shutdown();
      return 1;
    }
    state.setJointGroupPositions(jmg, seed);
  }

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

  auto emit_result = [&](const geometry_msgs::msg::Pose& pose) -> bool {
    bool ik_success = false;

    for (int attempt = 0; attempt < options.ik_attempts && !ik_success; ++attempt)
    {
      if (attempt > 0)
      {
        state.setToRandomPositions(jmg);
      }
      ik_success = state.setFromIK(jmg, pose, options.ee_link, options.ik_timeout);
    }

    bool self_collision_free = false;
    if (ik_success)
    {
      state.update();
      result.clear();
      scene->checkSelfCollision(request, result, state);
      self_collision_free = !result.collision;
    }

    std::vector<double> solution;
    solution.resize(jmg->getVariableCount(), 0.0);
    if (ik_success)
    {
      state.copyJointGroupPositions(jmg, solution);
    }

    (*output_stream) << (ik_success ? 1 : 0) << ' ' << (self_collision_free ? 1 : 0) << ' '
                     << ((ik_success && self_collision_free) ? 1 : 0);
    for (double value : solution)
    {
      (*output_stream) << ' ' << value;
    }
    (*output_stream) << '\n';

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
      if (values.size() != 7)
      {
        RCLCPP_ERROR(logger, "Pose size mismatch on line %zu", line_number);
        rclcpp::shutdown();
        return 1;
      }
      geometry_msgs::msg::Pose pose;
      pose.position.x = values[0];
      pose.position.y = values[1];
      pose.position.z = values[2];
      pose.orientation.x = values[3];
      pose.orientation.y = values[4];
      pose.orientation.z = values[5];
      pose.orientation.w = values[6];

      if (!emit_result(pose))
      {
        rclcpp::shutdown();
        return 1;
      }
    }
  }
  else
  {
    const std::vector<double> values = parseDoubleList(options.pose_csv);
    if (values.size() != 7)
    {
      RCLCPP_ERROR(logger, "Pose size mismatch: expected 7 values");
      rclcpp::shutdown();
      return 1;
    }
    geometry_msgs::msg::Pose pose;
    pose.position.x = values[0];
    pose.position.y = values[1];
    pose.position.z = values[2];
    pose.orientation.x = values[3];
    pose.orientation.y = values[4];
    pose.orientation.z = values[5];
    pose.orientation.w = values[6];

    if (!emit_result(pose))
    {
      rclcpp::shutdown();
      return 1;
    }
  }

  rclcpp::shutdown();
  return 0;
}
