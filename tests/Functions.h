// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef EXACT_H
#define EXACT_H
#include "headers/Utils.h"
#include "math.h"
#include "headers/Core/Definitions.h"

// Cavity (Check the Threashold)

inline Vec3 LID_CAVITY_FLOW(double x, double y,double z,double t){
    Vec3 r;
    r.u = 0.0;
    r.v = 0.0;
    r.w = 0.0;
    if(y > 1.0){
        r.u = 1.0;
    }

    return r;

}

inline double LID_CAVITY_FLOW_PRESSURE(double x, double y,double z,double t){

    return 0.0;

}

// Identity Functions

inline Vec3 ZERO(double x, double y,double z,double ){
    Vec3 vec;
    vec.u = 0;
    vec.v = 0;
    vec.w = 0;
    return vec;
}

inline Vec2 ZERO2D(double x, double y,double ){
    Vec2 vec;
    vec.u = 0;
    vec.v = 0;

    return vec;
}

inline double ZERO_SCALAR(double x,double y,double z,double t){
    return 0.0;
}

inline double ZERO2D_SCALAR(double x,double y,double t){
    return 0.0;
}

// Solid Masks

inline int LID_CAVITY_SOLID_MASK(int i,int j,int k){
    if(i == 0 || j == 0 || k == 0 || i == SIMULATION.Ny-1 ||  j == SIMULATION.Nx-1 || k == SIMULATION.Nz-1){
        return SOLID_CELL;
    }
    else{

        return FLUID_CELL;
    }
}

// Lid Cavity 2D

inline int LID_CAVITY_SOLID_MASK_2D(int i,int j){
    if(i == 0 || j == 0 ||  i == SIMULATION.Ny-1 ||  j == SIMULATION.Nx-1 ){
        return SOLID_CELL;
    }
    else{

        return FLUID_CELL;
    }
}

inline Vec2 LID_CAVITY_FLOW_2D(double x, double y,double t){
    Vec2 r;
    r.u = 0.0;
    r.v = 0.0;
    if(y > 1.0){
        r.u = 1.0;
    }

    return r;

}

inline double LID_CAVITY_FLOW_PRESSURE_2D(double x, double y,double t){

    return 0.0;

}

// OBSTACLE  3D
inline Vec3 OBSTACLE_FLOW(double x, double y,double z,double t){
    Vec3 r;
    r.u = 0.0;
    r.v = 0.0;
    r.w = 0.0;
    double U0 = 0.45;

    if(x <= SIMULATION.dh  && y < (6.0 - SIMULATION.dh/2.0) && y > 0.0 && z < (6.0 - SIMULATION.dh/2.0) && z > 0.0){
        r.u = 16.0*U0*y*z*(0.41 - y) *(0.41-z)/pow(0.41,4);

    }
    if(r.u <0){
        r.u = 0.0;
    }

    return r;

}

inline double  OBSTACLE_FLOW_PRESSURE(double x, double y,double z,double t){

    return 0.0;

}

inline int OBSTACLE_SOLID_MASK(int i, int j, int k) {
    double dh = SIMULATION.dh;
    double x = (j - 1) * dh;   // cell j's lower corner: the ring (index 0) lies outside the domain
    double y = (i - 1) * dh;

    // Check for inflow condition FIRST (most specific)
    if(j == 0 && i >= 1 && i < SIMULATION.Ny-1 && k != 0 && k != SIMULATION.Nz-1) {
        return INFLOW_CELL;
    }

    // Check for outflow/empty condition
    if(j == SIMULATION.Nx-1 && i != 0 && i != SIMULATION.Ny-1 && k != 0 && k != SIMULATION.Nz-1) {
        return EMPTY_CELL;
    }

    // Check for solid boundaries
    if(i == 0 || j == 0 || k == 0 || i == SIMULATION.Ny-1 || j == SIMULATION.Nx-1 || k == SIMULATION.Nz-1) {
        return SOLID_CELL;
    }

    double cx= 0.15;
    double cl = 0.1;

    double radius = 0.05;
    double center = 0.2;

    if(pow(((j - 1)*dh - center),2) +pow(((i - 1)*dh - center),2)  < radius*radius){
        return SOLID_CELL;
    }

    return FLUID_CELL;
}

// OBSTACLE  2D

inline Vec2 OBSTACLE_FLOW_2D(double x, double y,double t){
    Vec2 r;
    r.u = 0.0;
    r.v = 0.0;

    double U0 = 1.5;

    if(x <= SIMULATION.dh  && y < (6.0 - SIMULATION.dh/2.0) && y > 0.0){
        r.u = U0 - (U0/(0.205*0.205))*pow((y - 0.205),2);
    }
    if(r.u <0){
        r.u = 0.0;
    }

    return r;

}

inline double  OBSTACLE_FLOW_PRESSURE_2D(double x, double y,double t){

    return 0.0;

}

inline int OBSTACLE_SOLID_MASK_2D(int  i,int j){
     // First check for empty cell condition
     if(j == SIMULATION.Nx-1 && i != 0 && i != SIMULATION.Ny-1) {
        return EMPTY_CELL;
    }
    // Then check for inflow condition
    if(j == 0 && i >= 1 && i < SIMULATION.Ny-1 ) {
        return INFLOW_CELL;
    }
    // Finally check for solid boundaries
    if(i == 0 || j == 0 || i == SIMULATION.Ny-1 || j == SIMULATION.Nx-1 ) {
        return SOLID_CELL;
    }
    double dh = SIMULATION.dh;
    double radius = 0.05;
    double center = 0.2;

    if(pow(((j - 1)*dh - center),2) +pow(((i - 1)*dh - center),2)  < radius*radius){
        return SOLID_CELL;
    }

    return FLUID_CELL;

}

// DAMBREAK 2D

inline Vec2 DAMBREAK_FONT_2D(double x, double y,double t){
    Vec2 r;
    r.u = SIMULATION.f.v;

    return r;

}

inline Vec2 DAMBREAK_BORDER_2D(double x, double y,double t){
    Vec2 r;
    r.u = 0.0;
    r.v = 0.0;

    return r;

}

inline double DAMBREAK_PRESSURE_2D(double x, double y,double t){

    return 0.0;

}

inline int DAMBREAK_SOLID_MASK_2D(int i,int j){
    double dh = SIMULATION.dh;
    double radius = 0.05;

    if(i == 0  || j == 0  || i == SIMULATION.Ny-1 ||    j == SIMULATION.Nx-1){
        return SOLID_CELL;
    }
    else{
        return FLUID_CELL;
    }
}

// DAMBREAK 3D

inline Vec3 DAMBREAK_FONT(double x, double y, double z, double t){
    Vec3 r;
    r.u = 0.0;
    r.v = SIMULATION.f.v;
    r.w = 0.0;
    return r;
}

inline Vec3 DAMBREAK_BORDER(double x, double y, double z, double t){
    Vec3 r;
    r.u = 0.0;
    r.v = 0.0;
    r.w = 0.0;
    return r;
}

inline double DAMBREAK_PRESSURE(double x, double y, double z, double t){
    return 0.0;
}

inline int DAMBREAK_SOLID_MASK(int i, int j, int k){

    if(i == 0 || j == 0 || k == 0 || i == SIMULATION.Ny-1 || j == SIMULATION.Nx-1 || k == SIMULATION.Nz-1){
        return SOLID_CELL;
    }
    else{
        return FLUID_CELL;
    }
}

#endif