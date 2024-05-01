import sys, argparse, time, math, glm, random
from utils import FPSCounter, ShaderProgram, Camera
from body_taichi import NBodySystem

import taichi as ti

import numpy as np
import moderngl as mgl

import pygame

import imgui
import my_imgui.pygame_imgui as pygame_imgui

# -----------------------------------------------------------------------------------------------------------

class App:

    def __init__(self, screen_width=1600, screen_height=1200, use_opengl=True, max_fps=-1, nb_body=8, dt=0.005, eps=0.5):

        # screen
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.use_opengl = use_opengl

        # FPS
        self.lastTime = time.time()
        self.currentTime = time.time()
        self.fps = FPSCounter()
        self.max_fps = max_fps
        self.clock = pygame.time.Clock()
        self.delta_time = 0

        # bodies
        self.nb_body = nb_body
        self.dt = dt
        self.eps = eps

        self.p1_mass = 1.0
        self.nbody_system = NBodySystem(p1_mass=self.p1_mass, p2_mass=1.0, p3_mass=1.0, px_mass=1.0, nb_body=self.nb_body)
        self.nbody_system.init_bodies()

        self.cube = [(-1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (-1.0, -1.0, 1.0), (-1.0, 1.0, -1.0), (1.0, 1.0, -1.0), (1.0, -1.0, -1.0), (-1.0, -1.0, -1.0)]

        # scene
        self.paused = False
        self.clear = True
        self.show_cube = True
        
        self.cols = [(255,200,155), (155,255,200), (200,155,255), (255,155,155)]
        while len(self.cols) < self.nb_body:
            self.cols.append((random.randint(64, 255), random.randint(128, 255), random.randint(196, 255)))

        # camera
        self.cam_speed = 0.01
        self.cam_fov = 45.
        self.camera = Camera(self, fov=self.cam_fov, near=0.01, far=100., position=(0, 0, 3), speed=self.cam_speed, sensivity=0.07)

        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)

        self.forward = False
        self.backward = False
        self.right = False
        self.left = False
        self.up = False
        self.down = False
        self.zoom_less = False
        self.zoom_more = False
        self.mouse_x, self.mouse_y = 0, 0
        self.mouse_button_down = False

        # pygame window
        f = pygame.RESIZABLE
        if use_opengl:
            f |= pygame.DOUBLEBUF | pygame.OPENGL

        self.window = pygame.display.set_mode((self.screen_width, self.screen_height), flags=f)
        # in non opengl mode we blit directly on the screen window
        self.screen = self.window

        if use_opengl:
            # pg.draw on this surface. then this surface is converted into a texture
            # then this texture is sampled2D in the FS and rendered into the screen (which is a 2 triangles  => quad)
            self.display = pygame.Surface((self.screen_width, self.screen_height))
            self.screen = self.display

            # OpenGL context / options
            self.ctx = mgl.create_context()

            self.ctx.enable(flags=mgl.BLEND)

            quad = [
                # pos (x, y), uv coords (x, y)
                -1.0, 1.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 0.0,
                -1.0, -1.0, 0.0, 1.0,
                1.0, -1.0, 1.0, 1.0,
            ]

            quad_buffer = self.ctx.buffer(data=np.array(quad, dtype='f4'))

            self.all_shaders = ShaderProgram(self.ctx)
            self.screen_program = self.all_shaders.get_program("screen")

            self.vao = self.ctx.vertex_array(self.screen_program, [(quad_buffer, '2f 2f', 'vert', 'texcoord')])

            self.frame_tex = self.surface_to_texture(self.display)
            self.frame_tex.use(0)
            self.screen_program['tex'] = 0

            self.ctx.clear(color=(0.0, 0.0, 0.0))

            # imgui window options
            imgui.create_context()
            self.imgui_renderer = pygame_imgui.PygameRenderer()
            imgui.get_io().display_size = self.screen_width, self.screen_height
            
    def surface_to_texture(self, surf):
        tex = self.ctx.texture(surf.get_size(), 4)
        tex.filter = (mgl.NEAREST, mgl.NEAREST)
        tex.swizzle = 'BGRA'
        # tex.write(surf.get_view('1'))
        return tex

    def get_fps(self):
        self.currentTime = time.time()
        delta = self.currentTime - self.lastTime

        if delta >= 1:
            gl_mode = "Non OpenGL"
            if self.use_opengl:
                gl_mode = "OpenGL"

            fps = f"FPS: {self.fps.get_fps():3.0f} ({gl_mode})"
            cam_pos = f"CamPos: {int(self.camera.position.x)}, {int(self.camera.position.y)}, {int(self.camera.position.z)}"
            pygame.display.set_caption(fps + " | " + cam_pos)

            self.lastTime = self.currentTime

        self.fps.tick()

    def show_options_ui(self):
        imgui.new_frame()
        imgui.begin("Options", True)

        _, self.show_cube = imgui.checkbox("Show Cube", self.show_cube)
        _, self.paused = imgui.checkbox("Pause (P key)", self.paused)
        _, self.clear = imgui.checkbox("Clear (C key)", self.clear)

        _, self.p1_mass  = imgui.slider_float("P1 mass", self.p1_mass, 0.01, 100.)
        self.nbody_system.bodies[0].mass = self.p1_mass

        _, self.dt  = imgui.slider_float("dt", self.dt, 0.00005, 0.01, format="%.5f")
        _, self.eps = imgui.slider_float("eps", self.eps, 0.01, 0.5)
        _, self.cam_speed = imgui.slider_float("cam_speed", self.cam_speed, 0.001, 1.0, format="%.3f")
        _, self.cam_fov = imgui.slider_float("cam_fov", self.cam_fov, 1.0, 90.0)

        self.camera.speed = self.cam_speed
        self.camera.fov = self.cam_fov

        imgui.end()

    def run(self):

        while True:

            if self.clear:
                self.screen.fill((0,0,0))

            self.zoom_more = False
            self.zoom_less = False

            # pygame events
            for event in pygame.event.get():

                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_p:
                        self.paused = not self.paused
                    if event.key == pygame.K_c:
                        self.clear = not self.clear

                    if event.key == pygame.K_UP:
                        self.forward = True
                    if event.key == pygame.K_DOWN:
                        self.backward = True
                    if event.key == pygame.K_RIGHT:
                        self.right = True
                    if event.key == pygame.K_LEFT:
                        self.left = True
                    if event.key == pygame.K_LCTRL:
                        self.up = True
                    if event.key == pygame.K_LSHIFT:
                        self.down = True
                    
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_UP:
                        self.forward = False
                    if event.key == pygame.K_DOWN:
                        self.backward = False
                    if event.key == pygame.K_RIGHT:
                        self.right = False
                    if event.key == pygame.K_LEFT:
                        self.left = False
                    if event.key == pygame.K_LCTRL:
                        self.up = False
                    if event.key == pygame.K_LSHIFT:
                        self.down = False

                    if event.key == pygame.K_q:
                        self.zoom_more = True
                    if event.key == pygame.K_w:
                        self.zoom_less = True

                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.mouse_button_down = True

                if event.type == pygame.MOUSEBUTTONUP:
                    self.mouse_button_down = False

                # imgui events
                if self.use_opengl:
                    self.imgui_renderer.process_event(event)

            # mouse camera control
            if self.mouse_button_down:
                mx, my = pygame.mouse.get_pos()

                if self.mouse_x:
                    self.mouse_dx = self.mouse_x - mx
                else:
                    self.mouse_dx = 0

                if self.mouse_y:
                    self.mouse_dy = self.mouse_y - my
                else:
                    self.mouse_dy = 0

                self.mouse_x = mx
                self.mouse_y = my

            else:
                self.mouse_x = 0
                self.mouse_y = 0
                self.mouse_dx, self.mouse_dy = 0, 0
                
            # body point size
            if self.clear:
                point_size = 8
            else:
                point_size = 1

            # particules positions
            if not self.paused:
                self.nbody_system.update(self.dt, self.eps)

            # camera motion
            self.camera.update(self.mouse_dx, self.mouse_dy, self.forward, self.backward, self.left, self.right, self.up, self.down, self.zoom_less, self.zoom_more)
       
            # move and put obj into our world
            self.obj_rot   = glm.vec3([glm.radians(a) for a in (0, 0, 0)])
            self.obj_scale = (1, 1, 1) # same size
            self.obj_pos   = (0, 0, 0) # into to center of our world

            # first do scaling operations, then rotations and lastly translations when combining matrices
            self.m_model = glm.mat4()
            self.m_model = glm.translate(self.m_model, self.obj_pos)
            self.m_model = glm.rotate(self.m_model, self.obj_rot.z, glm.vec3(0, 0, 1))
            self.m_model = glm.rotate(self.m_model, self.obj_rot.y, glm.vec3(0, 1, 0))
            self.m_model = glm.rotate(self.m_model, self.obj_rot.x, glm.vec3(1, 0, 0))
            self.m_model = glm.scale (self.m_model, self.obj_scale)

            if self.show_cube:

                cube_points = []
                for point in self.cube:

                    x_ndc = point[0] # [-1, 1]
                    y_ndc = point[1] # [-1, 1]
                    z_ndc = point[2] # [-1, 1]

                    ndc_hmg_vec4 = glm.vec4(x_ndc, y_ndc, z_ndc, 1.0) # (pos, w) w = hmg coord for mat4x4 translation / projection ops
                    pt_proj_vec4 = self.camera.get_projection_matrix() * self.camera.get_view_matrix() * self.m_model * ndc_hmg_vec4

                    x_proj = (self.screen_width/2) +  (pt_proj_vec4.x/pt_proj_vec4.z)*(self.screen_width/2)
                    y_proj = (self.screen_height/2) - (pt_proj_vec4.y/pt_proj_vec4.z)*(self.screen_height/2)
                    
                    cube_points.append( (int(x_proj), int(y_proj)) )

                # render cube
                col = (96, 96, 96)
                pygame.draw.line(self.screen, col, cube_points[0], cube_points[1], width=1)
                pygame.draw.line(self.screen, col, cube_points[1], cube_points[2], width=1)
                pygame.draw.line(self.screen, col, cube_points[2], cube_points[3], width=1)
                pygame.draw.line(self.screen, col, cube_points[3], cube_points[0], width=1)
                pygame.draw.line(self.screen, col, cube_points[4], cube_points[5], width=1)
                pygame.draw.line(self.screen, col, cube_points[5], cube_points[6], width=1)
                pygame.draw.line(self.screen, col, cube_points[6], cube_points[7], width=1)
                pygame.draw.line(self.screen, col, cube_points[7], cube_points[4], width=1)
                pygame.draw.line(self.screen, col, cube_points[0], cube_points[4], width=1)
                pygame.draw.line(self.screen, col, cube_points[1], cube_points[5], width=1)
                pygame.draw.line(self.screen, col, cube_points[2], cube_points[6], width=1)
                pygame.draw.line(self.screen, col, cube_points[3], cube_points[7], width=1)

            # Particules
            bodies = []
            for i in range(self.nb_body):
                
                x_ndc = self.nbody_system.bodies[i].pos.x
                y_ndc = self.nbody_system.bodies[i].pos.y
                z_ndc = self.nbody_system.bodies[i].pos.z
            
                ndc_hmg_vec4 = glm.vec4(x_ndc, y_ndc, z_ndc, 1.0)
                pt_proj_vec4 = self.camera.get_projection_matrix() * self.camera.get_view_matrix() * self.m_model * ndc_hmg_vec4

                x_proj = (self.screen_width/2) +  (pt_proj_vec4.x/pt_proj_vec4.z)*(self.screen_width/2)
                y_proj = (self.screen_height/2) - (pt_proj_vec4.y/pt_proj_vec4.z)*(self.screen_height/2)

                bodies.append( (int(x_proj), int(y_proj)) )
                
            # render particules
            for i, b in enumerate(bodies):
                pygame.draw.circle(self.screen, self.cols[i], (b[0],  b[1]), point_size)

            # opengl mode => write tour 2D pygame surface into the texture (which will be be rendered in a quad by the fragment shader)
            if self.use_opengl:
                try:
                    self.frame_tex.write(self.display.get_view('1'))
                    #self.frame_tex.write(self.display.get_buffer())
                except:
                    pass

                self.vao.render(mode=mgl.TRIANGLE_STRIP)

                if self.use_opengl:
                    self.show_options_ui()
                    imgui.render()
                    self.imgui_renderer.render(imgui.get_draw_data())
                
            # display
            pygame.display.flip()

            # fps
            self.delta_time = self.clock.tick(self.max_fps)
            self.get_fps()

# -----------------------------------------------------------------------------------------------------------
# python3 main.py --arch=cpu --body=8 --fps=-1
# python3 main.py --arch=vulkan --body=8 --fps=60

def main():

    pygame.init()
    pygame.mouse.set_visible(True)
    pygame.font.init()

    # const
    USE_PROFILER  = 0

    SCREEN_WIDTH  = 1280
    SCREEN_HEIGHT = 800

    # args
    parser = argparse.ArgumentParser(description="Leapfrog N-Body")

    parser.add_argument('-a', '--arch', help='Taichi backend', default="cpu", action="store")
    parser.add_argument('-f', '--fps', help='Max FPS, -1 for unlimited', default=-1, type=int)
    parser.add_argument('-b', '--body', help='NB Body', default=3, type=int)

    result = parser.parse_args()
    args = dict(result._get_kwargs())

    print("Args = %s" % args)

    if args["arch"] in ("cpu", "x64"):
        ti.init(ti.cpu, debug=0, default_ip=ti.i32, default_fp=ti.f32, kernel_profiler=USE_PROFILER)

    elif args["arch"] in ("gpu", "cuda"):
        ti.init(ti.gpu, kernel_profiler=USE_PROFILER)
    elif args["arch"] in ("opengl",):
        ti.init(ti.opengl)
    elif args["arch"] in ("vulkan",):
        ti.init(ti.vulkan)

    # App
    app = App(screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT, use_opengl=1, max_fps=args["fps"], 
              nb_body=args["body"], dt=0.0005, eps=0.1)
    app.run()

if __name__ == "__main__":
    main()