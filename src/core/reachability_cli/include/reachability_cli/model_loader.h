// File: model_loader.h
// Purpose: Declare helpers for loading MoveIt robot models and planning scenes from URDF and SRDF files.
// Usage: Included by reachability_cli executables and supporting libraries.

#ifndef REACHABILITY_CLI__MODEL_LOADER_H
#define REACHABILITY_CLI__MODEL_LOADER_H

#include <memory>
#include <string>

#include <moveit/robot_model/robot_model.h>
#include <moveit/planning_scene/planning_scene.h>

namespace reachability_cli
{

struct RobotContext
{
  moveit::core::RobotModelPtr model;
  planning_scene::PlanningScenePtr scene;
};

RobotContext loadRobotFromFiles(const std::string& urdf_path, const std::string& srdf_path);

}  // namespace reachability_cli

#endif  // REACHABILITY_CLI__MODEL_LOADER_H
