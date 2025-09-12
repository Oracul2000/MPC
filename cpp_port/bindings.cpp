#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // для автоматического преобразования типов STL
#include "main.cpp" // Включаем основной код модели

namespace py = pybind11;

PYBIND11_MODULE(car_model, m) {
    m.doc() = "pybind11 plugin for a 4-wheel vehicle dynamics model";

    // Привязка структуры controlInfluence
    py::class_<controlInfluence>(m, "ControlInfluence")
        .def(py::init<double, double, double>(), py::arg("throttle"), py::arg("steeringAngle"), py::arg("brakes"))
        .def_readwrite("throttle", &controlInfluence::throttle)
        .def_readwrite("steeringAngle", &controlInfluence::steeringAngle)
        .def_readwrite("brakes", &controlInfluence::brakes)
        .def("__repr__",
            [](const controlInfluence &c) {
                return "<ControlInfluence throttle=" + std::to_string(c.throttle) +
                       ", steering=" + std::to_string(c.steeringAngle) +
                       ", brakes=" + std::to_string(c.brakes) + ">";
            }
        );

    // Привязка структуры chassis
    py::class_<chassis>(m, "Chassis")
        .def(py::init<>())
        .def_readonly("X", &chassis::X)
        .def_readonly("Y", &chassis::Y)
        .def_readonly("yaw", &chassis::yaw)
        .def_readonly("vx", &chassis::vx)
        .def_readonly("vy", &chassis::vy)
        .def_readonly("w", &chassis::w)
        .def("__repr__",
            [](const chassis &c) {
                return "<Chassis X=" + std::to_string(c.X) +
                       ", Y=" + std::to_string(c.Y) +
                       ", yaw=" + std::to_string(c.yaw) +
                       ", vx=" + std::to_string(c.vx) +
                       ", vy=" + std::to_string(c.vy) +
                       ", w=" + std::to_string(c.w) + ">";
            }
        );

    // Привязка основного класса модели
    py::class_<Dynamic4WheelsModel>(m, "Dynamic4WheelsModel")
        .def(py::init<>())
        .def("update", &Dynamic4WheelsModel::updateRK4, "Update vehicle state using RK4 integration for one time step (dt)", py::arg("control"))
        .def("get_state", &Dynamic4WheelsModel::getChassisState, "Get the current state of the vehicle chassis")
        .def_property_readonly("t", [](const Dynamic4WheelsModel &model) { return model.t; })
        .def("set_initial_state", &Dynamic4WheelsModel::set_initial_state, "Set the initial state of the vehicle", 
             py::arg("x") = 0.0, py::arg("y") = 0.0, py::arg("yaw") = 0.0, 
             py::arg("vx") = 0.0, py::arg("vy") = 0.0, py::arg("w") = 0.0)
        .def("__repr__",
            [](const Dynamic4WheelsModel &model) {
                 const auto& body = model.carState.body;
                 return "<Dynamic4WheelsModel t=" + std::to_string(model.t) +
                        " | pos=(" + std::to_string(body.X) + ", " + std::to_string(body.Y) +
                        "), vel=(" + std::to_string(body.vx) + ", " + std::to_string(body.vy) + ")>";
            }
        );
}