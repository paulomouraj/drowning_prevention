# WaterPi, a software to prevent drowning in home swimming pools

## Description

The goal of this work was to develop a software package for detecting pedestrians walking too close to the borders of home swimming pools that could fall off the border and drown without being noticed. The software runs in a Raspberry Pi 3 with a Pi Camera module and thus was designed to work with limited hardware. The final tests performed with children revealed a sucess ratio of 87% in 46 approximations to the pool in different ways.

A detection example is shown below:
![Example of detection in the pool area](https://github.com/paulomouraj/waterpi_drowning_prevention/blob/main/examples/detection1.jpg)

The detection pipeline is based on the following steps:

1. Pool area detection: The user must press a button to calibrate the pool area. The system will identify the pool area by the color of the borders.
2. Movement detection: The system will use a MOG2 background subtractor to detect movement in the frames.
3. Person detection: The system will use a pre-trained MobileNet SSD CNN model to detect people in the frame. If a person is detected, the system will check if the person is close to the pool area and an alarm could be sounded.

## Models

The model CNN object detection with bounding boxes: [MobileNet-SSD v2](https://github.com/chuanqi305/MobileNet-SSD/tree/master)

## Hardware Requirements

- Raspberry Pi 3
- Raspberry Pi Camera Module v2.1

## Software Requirements

- Python 3.8.0
- NumPy 1.19.2
- Picamera 1.13
- OpenCV 4.1.1

## Contributors

Paulo Roberto de Moura Júnior (me)  
Kevin Mulinari Kuhn

[High level description (only in portuguese)](https://github.com/paulomouraj/waterpi_drowning_prevention/blob/main/docs/waterpi_HLD.pdf)
