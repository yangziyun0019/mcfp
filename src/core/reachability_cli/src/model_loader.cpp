// File: model_loader.cpp
// Purpose: Build MoveIt robot and planning-scene objects directly from URDF and SRDF files.
// Usage: Linked by reachability_cli executables through loadRobotFromFiles().

#include "reachability_cli/model_loader.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>

namespace reachability_cli
{

namespace
{
std::string readTextFile(const std::string& path)
{
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file)
  {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}
}  // namespace

RobotContext loadRobotFromFiles(const std::string& urdf_path, const std::string& srdf_path)
{
  const std::string urdf_xml = readTextFile(urdf_path);

  urdf::ModelInterfaceSharedPtr urdf_model = urdf::parseURDF(urdf_xml);
  if (!urdf_model)
  {
    throw std::runtime_error("Failed to parse URDF: " + urdf_path);
  }

  auto srdf_model = std::make_shared<srdf::Model>();
  if (!srdf_model->initFile(*urdf_model, srdf_path))
  {
    throw std::runtime_error("Failed to parse SRDF: " + srdf_path);
  }

  auto robot_model = std::make_shared<moveit::core::RobotModel>(urdf_model, srdf_model);
  if (robot_model->isEmpty())
  {
    throw std::runtime_error("Robot model is empty after loading URDF/SRDF");
  }

  auto scene = std::make_shared<planning_scene::PlanningScene>(robot_model);

  RobotContext context;
  context.model = robot_model;
  context.scene = scene;
  return context;
}

}  // namespace reachability_cli
