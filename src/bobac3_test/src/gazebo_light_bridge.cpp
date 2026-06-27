#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <gazebo/gazebo_client.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <ignition/math/Color.hh>

gazebo::transport::PublisherPtr lightPub;
std::string light_name = "bobac3_serverbot::cover_link::top_led_light";

void cmdCallback(const std_msgs::Bool::ConstPtr& msg) {
    gazebo::msgs::Light light_msg;
    light_msg.set_name(light_name);
    if (msg->data) {
        // Red light
        gazebo::msgs::Set(light_msg.mutable_diffuse(), ignition::math::Color(1.0, 0.0, 0.0, 1.0));
        gazebo::msgs::Set(light_msg.mutable_specular(), ignition::math::Color(1.0, 0.0, 0.0, 1.0));
    } else {
        // Turn off
        gazebo::msgs::Set(light_msg.mutable_diffuse(), ignition::math::Color(0.0, 0.0, 0.0, 0.0));
        gazebo::msgs::Set(light_msg.mutable_specular(), ignition::math::Color(0.0, 0.0, 0.0, 0.0));
    }
    lightPub->Publish(light_msg);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "gazebo_light_bridge");
    ros::NodeHandle nh;

    // Load gazebo client to be able to use transport
    gazebo::client::setup(argc, argv);
    gazebo::transport::NodePtr node(new gazebo::transport::Node());
    node->Init();

    lightPub = node->Advertise<gazebo::msgs::Light>("~/light/modify");

    ros::Subscriber sub = nh.subscribe("/top_led/cmd", 1, cmdCallback);
    
    ROS_INFO("Gazebo light bridge started. Listening on /top_led/cmd");
    ros::spin();

    gazebo::client::shutdown();
    return 0;
}
