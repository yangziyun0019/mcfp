//
// Created by yezi on 2025/5/4.
//

#include <arx_hardware_interface/ARXJoy.hpp>
#include <arx_hardware_interface/canbase/SocketCan.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>

using namespace arx;
using namespace hw_interface;

ARXJoy joy;
void receiveCB(CanFrame *frame) { joy.read(frame); }

void exchangeCanMsg() { joy.update(); }

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("arx_joy");
  auto button_pub =
      node->create_publisher<std_msgs::msg::Int32MultiArray>("/arx_joy", 10);
  SocketCan can;
  can.setCallBackFunction(receiveCB);
  can.setGetMsgContentFunction(exchangeCanMsg);
  can.Init("can6");
  can.closeReset();
  rclcpp::Rate rate(100);
  while (rclcpp::ok()) {
    int button[8];
    can.LoadMutexMsg();
    joy.getValue(button);
    std_msgs::msg::Int32MultiArray msg;
    for (int i = 0; i < 4; i++)
      msg.data.push_back(button[i]);
    button_pub->publish(msg);
    rate.sleep();
  }
  return 0;
}
