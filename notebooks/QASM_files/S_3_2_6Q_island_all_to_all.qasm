OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
cx q[0],q[1];
u3(pi/2,0,-pi) q[0];
u3(pi/2,-2.0599179,-pi/2) q[1];
u3(1.5006511,-pi/2,-pi) q[2];
cx q[1],q[2];
u3(3.1091932,-pi,-pi/2) q[1];
u3(1.5597012,1.3547106,-1.6730006) q[2];
cx q[1],q[2];
u3(pi/2,pi/2,-0.25129896) q[1];
u3(2.5767194,-pi,-pi/2) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
cx q[0],q[3];
cx q[1],q[4];
cx q[2],q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5];
u3(1.5473325,-2.1220976,-2.8283795) q[0];
u3(0.59689723,-2.1542102,0.37275859) q[1];
cx q[0],q[1];
u3(0.80970734,-pi/2,pi/2) q[0];
u3(0.54145008,0.58497791,0.91282857) q[1];
cx q[0],q[1];
u3(0.049518588,0,-pi/2) q[0];
u3(1.9692524,2.0557939,-2.9399067) q[1];
cx q[0],q[1];
u3(1.1321279,2.8927483,-2.3357534) q[0];
cx q[0],q[2];
u3(1.9524113,2.4434647,-0.74053797) q[1];
u3(0,0,pi/4) q[2];
cx q[1],q[2];
u3(0,0,5.86372175139512) q[2];
cx q[0],q[2];
u3(0.52701022,-pi,2.4222942) q[0];
u3(0,0,4.29292542460023) q[2];
cx q[1],q[2];
u3(0.93414707,1.1961224,-2.8579441) q[1];
cx q[0],q[1];
u3(2.6363526,-pi,-pi/2) q[0];
u3(1.608383,1.5107846,-0.45186088) q[1];
cx q[0],q[1];
u3(1.2899306,2.6138753,-pi/2) q[0];
u3(1.1986411,-1.6579064,1.282879) q[1];
u3(0,0,5*pi/4) q[2];
cx q[2],q[0];
u3(pi/2,3*pi/4,-pi) q[0];
cx q[1],q[0];
u3(pi/2,0,-3*pi/4) q[0];
u3(0.71883,-pi/2,-pi/2) q[2];
cx q[0],q[2];
u3(3*pi/4,2.5261129,-pi/2) q[0];
u3(1.6372168,1.5040808,-0.0044345716) q[2];
cx q[0],q[2];
u3(2.2555155,-0.42053434,pi/2) q[0];
u3(1.6373645,-pi/4,pi/2) q[2];
cx q[1],q[2];
cx q[1],q[0];
u3(2.0344439,3*pi/4,pi/2) q[0];
u3(0,0,pi/4) q[2];
u3(-pi/2,-pi/2,pi/2) q[3];
u3(1.8856349,pi/2,0) q[4];
cx q[3],q[4];
u3(pi,-pi/2,-2.9855492) q[3];
u3(0,0,1.01318062365737) q[4];
cx q[5],q[4];
u3(1.8738865,0.089847394,pi/2) q[4];
cx q[5],q[3];
u3(pi/2,pi/2,-2.9855492) q[3];
cx q[5],q[4];
u3(-0.626912531032031,-pi/2,pi/2) q[4];
cx q[4],q[3];
u3(0,0,7.70615653968503) q[3];
u3(pi/2,-pi/4,0.21892176) q[4];
u3(pi/2,-pi/2,-pi) q[5];
cx q[3],q[5];
u3(pi,-pi/2,pi/2) q[3];
u3(0,0,-pi/4) q[5];
cx q[4],q[5];
u3(1.7792616,pi/2,pi/2) q[4];
u3(2.11970787168946,-pi/2,pi/2) q[5];
cx q[3],q[5];
u3(pi,-pi/2,0) q[3];
u3(1.6039883,1.5131255,-0.5264754) q[5];
cx q[3],q[5];
u3(pi,-pi/2,-pi/2) q[3];
cx q[3],q[4];
u3(pi,-1.0418141,-pi/2) q[3];
u3(0.80682066,1.2801098,0.20407752) q[4];
u3(1.1087857,-0.52123956,-0.66011138) q[5];
cx q[4],q[5];
u3(1.5237429,-0.78428952,3.0944871) q[4];
u3(pi/2,0,-pi) q[5];
cx q[3],q[5];
u3(-pi/2,-pi/2,pi/2) q[3];
cx q[3],q[4];
u3(pi/2,-pi,-pi/2) q[3];
u3(1.6372168,1.5040808,-0.0044345716) q[4];
cx q[3],q[4];
u3(pi/2,3*pi/4,-pi/2) q[3];
u3(3.0750245,-0.40424099,-pi/2) q[4];
u3(0,0,3*pi/2) q[5];
