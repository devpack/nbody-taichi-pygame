# nbody-taichi-pygame

Python-Taichi version of the nbody problem. Taichi is used for the N² computation (no BH tree used) and PyGame handles the draw calls into a regular pygame surface (3D is done by hand for the matrix fun), then this surface is converted into a texture which is rendered in a quad (ModernGL window) by the fragment shader. Options can be adjusted on the fly using Dear Imgui bindings.

Leapfrog / Verlet integration has been used for its energy conservation property.

----

Usage:

```
python3 main.py --arch=cpu --body=3 --fps=-1
```

```
Key C: clear ON/OFF
Key P: pause ON/OFF
Arrows and mouse to move the camera
Left-Shift = move up
Left-Control = move down
Q: zoom in
W: zoom out
```

----

![1](https://github.com/devpack/nbody-taichi-pygame/blob/main/assets/wiki/1.png)

![2](https://github.com/devpack/nbody-taichi-pygame/blob/main/assets/wiki/2.png)

![3](https://github.com/devpack/nbody-taichi-pygame/blob/main/assets/wiki/3.png)
