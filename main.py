import sys, argparse, time, math, glm, random
from utils import FPSCounter, ShaderProgram, Camera, ScreenRecorder
from body_taichi import NBodySystem

import taichi as ti

import numpy as np
import moderngl as mgl

import pygame, collections

import imgui
import my_imgui.pygame_imgui as pygame_imgui

# -----------------------------------------------------------------------------------------------------------

class App:

    def __init__(self, screen_width=1280, screen_height=800, use_opengl=True, orbital_mode=1, record_video="", video_fps=60, max_fps=-1, 
                 nb_body=8, use_taichi_for_matrix=False, dt=0.005, eps=0.5):

        # screen
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.use_opengl = use_opengl
        self.record_video = record_video
        self.video_fps = video_fps
        self.use_taichi_for_matrix = use_taichi_for_matrix

        if self.record_video:
            if self.record_video in ("XVID", "h264", "avc1", "mp4v"):
                self.video_recorder = ScreenRecorder(self.screen_width, self.screen_height, self.video_fps, codec=self.record_video)
            else:
                self.video_recorder = ScreenRecorder(self.screen_width, self.screen_height, self.video_fps)

        # FPS
        self.lastTime = time.time()
        self.currentTime = time.time()
        self.fps = FPSCounter()
        self.max_fps = max_fps
        self.clock = pygame.time.Clock()
        self.delta_time = 0
        self.frames = 0

        # bodies
        self.nb_body = nb_body
        self.dt = dt
        self.eps = eps
        self.p1_mass = 1.0
        self.trace_lenght = 1000
        self.point_size = 1
        self.fake_g = 2

        self.trace_deque = collections.deque(maxlen=self.trace_lenght*self.nb_body)
        self.trace_proj = []

        self.nbody_system = NBodySystem(screen_width=self.screen_width, screen_height=self.screen_height, fov=45, near=0.01, far=100., 
                                        p1_mass=self.p1_mass, p2_mass=1.0, p3_mass=1.0, px_mass=1.0, nb_body=self.nb_body)
        self.nbody_system.init_bodies()
        #self.nbody_system.init_bodies_3()
        #self.nbody_system.init_bodies_2()

        #self.field_points = self.get_field(min=-1.0, max=1.0, ppl=self.field_size)

        self.cube = [(-1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (-1.0, -1.0, 1.0), (-1.0, 1.0, -1.0), (1.0, 1.0, -1.0), (1.0, -1.0, -1.0), (-1.0, -1.0, -1.0)]

        # scene
        self.paused = False
        self.clear = True
        self.show_cube = True
        self.light = False
        self.light_att = 16
        self.glare = 196

        self.cols = [(255, 196, 128), (128, 255, 196), (196, 128, 255), (255, 128, 196), (128, 196, 255), (196, 255, 128)]
        while len(self.cols) < self.nb_body:
            self.cols.append((random.randint(64, 255), random.randint(128, 255), random.randint(196, 255)))

        # camera
        self.cam_speed = 0.01
        self.cam_fov = 48.
        self.orbital_mode = orbital_mode
        self.orbital_speed = 0.1
        self.camera = Camera(self, orbital_mode=self.orbital_mode, orbital_speed=self.orbital_speed, fov=self.cam_fov, near=0.01, far=100., position=(0.0, 0.5, 4), speed=self.cam_speed, sensivity=0.07)

        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)

        self.forward = False
        self.backward = False
        self.right = False
        self.left = False
        self.up = False
        self.down = False
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
            ft = "frames=%s" % str(self.frames)
            if self.use_taichi_for_matrix:
                matrix = "gpu matrix" 
            else:
                matrix = "cpu matrix" 
            pygame.display.set_caption(fps + " | " + matrix + " | " + cam_pos + " | " + ft)

            self.lastTime = self.currentTime

        self.fps.tick()

    def show_options_ui(self):
        imgui.new_frame()
        imgui.begin("Options", True)

        _, self.paused = imgui.checkbox("Pause (P key)", self.paused)
        _, self.clear = imgui.checkbox("Clear (C key)", self.clear)
        _, self.show_cube = imgui.checkbox("Show Cube", self.show_cube)

        _, self.light = imgui.checkbox("Light", self.light)
        _, self.light_att  = imgui.slider_int("Light att", self.light_att, 1, 64)
        _, self.glare  = imgui.slider_int("Glare", self.glare, 64, 255)

        _, self.orbital_mode = imgui.checkbox("Orbital Mode", self.orbital_mode)
        if _:
            self.camera.orbital_mode = self.orbital_mode
            if not self.orbital_mode:
                self.camera.position = glm.vec3(0, 0, 3.5)
        _, self.orbital_speed  = imgui.slider_float("Orbital Speed", self.orbital_speed, 0.001, 1.)
        if _:
            self.camera.orbital_speed = self.orbital_speed

        _, self.point_size  = imgui.slider_int("point size", self.point_size, 1, 8)

        _, self.trace_lenght  = imgui.slider_int("trace lenght", self.trace_lenght, 1, 3000)
        if _:
            self.trace_deque = collections.deque(self.trace_deque, maxlen=self.trace_lenght*self.nb_body)

        _, self.p1_mass  = imgui.slider_float("P1 mass", self.p1_mass, 0.01, 100.)
        if _:
            self.nbody_system.bodies[0].mass = self.p1_mass

        _, self.dt = imgui.slider_float("dt", self.dt, 0.00005, 0.01, format="%.5f")
        _, self.fake_g = imgui.slider_float("G", self.fake_g, 0.1, 128, format="%.1f")

        _, self.eps = imgui.slider_float("eps", self.eps, 0.01, 0.5)
        _, self.cam_speed = imgui.slider_float("cam speed", self.cam_speed, 0.001, 1.0, format="%.3f")
        _, self.cam_fov = imgui.slider_float("cam fov", self.cam_fov, 1.0, 90.0)
        if _:
            self.camera.speed = self.cam_speed
            self.camera.fov = self.cam_fov

        imgui.end()

    def draw_cube(self):
        # cube
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
            V = 64
            col=(V, V, V); B=(0, 0, V); R=(V, 0, 0) ; G = (0, V, 0)
            pygame.draw.line(self.screen, col, cube_points[0], cube_points[1], width=1)
            pygame.draw.line(self.screen, col, cube_points[1], cube_points[2], width=1)
            pygame.draw.line(self.screen, R,   cube_points[2], cube_points[3], width=1)
            pygame.draw.line(self.screen, G,   cube_points[3], cube_points[0], width=1)
            pygame.draw.line(self.screen, col, cube_points[4], cube_points[5], width=1)
            pygame.draw.line(self.screen, col, cube_points[5], cube_points[6], width=1)
            pygame.draw.line(self.screen, col, cube_points[6], cube_points[7], width=1)
            pygame.draw.line(self.screen, col, cube_points[7], cube_points[4], width=1)
            pygame.draw.line(self.screen, col, cube_points[0], cube_points[4], width=1)
            pygame.draw.line(self.screen, col, cube_points[1], cube_points[5], width=1)
            pygame.draw.line(self.screen, col, cube_points[2], cube_points[6], width=1)
            pygame.draw.line(self.screen, B,   cube_points[3], cube_points[7], width=1)

    def run(self):

        while True:

            if self.clear:
                self.screen.fill((0,0,0))

            # pygame events
            for event in pygame.event.get():

                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    if self.record_video:
                        self.video_recorder.end_recording()
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

            # calc particules positions
            if not self.paused:
                self.nbody_system.update_O2(self.dt, self.eps, self.fake_g)
                #self.nbody_system.update(self.dt, self.eps, self.fake_g)

            # camera motion
            self.camera.update(self.mouse_dx, self.mouse_dy, self.forward, self.backward, self.left, self.right, self.up, self.down)
       
            # move and put obj into our world
            self.obj_rot   = glm.vec3([glm.radians(a) for a in (0, 0, 0)])
            self.obj_scale = (1, 1, 1) # same size
            self.obj_pos   = (0, 0, 0) # into to center of our world

            # model mat4x4
            self.m_model = glm.mat4()
            self.m_model = glm.translate(self.m_model, self.obj_pos)
            self.m_model = glm.rotate(self.m_model, self.obj_rot.z, glm.vec3(0, 0, 1))
            self.m_model = glm.rotate(self.m_model, self.obj_rot.y, glm.vec3(0, 1, 0))
            self.m_model = glm.rotate(self.m_model, self.obj_rot.x, glm.vec3(1, 0, 0))
            self.m_model = glm.scale (self.m_model, self.obj_scale)

            self.draw_cube()

            # --- calc particules world positions

            # matrix multiplication on the GPU side vs CPU side

            if not self.paused:
                for i in range(self.nb_body):
                    self.trace_deque.append((self.nbody_system.bodies[i].pos.x, self.nbody_system.bodies[i].pos.y, self.nbody_system.bodies[i].pos.z))

            self.trace_proj = []

            if not self.use_taichi_for_matrix:
                for b in self.trace_deque:
                    
                    x_ndc = b[0]
                    y_ndc = b[1]
                    z_ndc = b[2]
                
                    pt_proj_vec4 = self.camera.get_projection_matrix() * self.camera.get_view_matrix() * self.m_model * glm.vec4(x_ndc, y_ndc, z_ndc, 1.0)

                    x_proj = (self.screen_width/2) + (pt_proj_vec4.x/pt_proj_vec4.w)*(self.screen_width/2)
                    y_proj = (self.screen_height/2) - (pt_proj_vec4.y/pt_proj_vec4.w)*(self.screen_height/2)

                    self.trace_proj.append( (x_proj, y_proj, pt_proj_vec4.w) )

            # tachi matrix ops
            else:
                #self.nbody_system.project_points_from_ti_bodies(self.camera.get_projection_matrix(), self.camera.get_view_matrix(), self.m_model)
                self.nbody_system.project_points_from_deque(self.camera.get_projection_matrix(), self.camera.get_view_matrix(), self.m_model, np.array(self.trace_deque))

                for i in range(len(self.trace_deque)):
                    self.trace_proj.append(self.nbody_system.projected_points[i])

            # render particules
            for i, body in enumerate(self.trace_proj):
                col = self.cols[i%self.nb_body]

                v = ( (i/self.nb_body) / (float(len(self.trace_proj))/self.nb_body) ) * self.glare
                if self.light:
                    z = body[2]
                    v -= z * self.light_att

                #r = 255 - max(0, min(255, col[0] - v))
                #g = 255 - max(0, min(255, col[1] - v))
                #b = 255 - max(0, min(255, col[2] - v))
                #col=(r, g, b)
                col = tuple(map(lambda x: 255 - max(0, min(255, x - v)), col))
                #if (i == len(self.trace_proj)-1) or (i == len(self.trace_proj)-self.nb_body) or (i == len(self.trace_proj)-(self.nb_body-1)*2) or \
                #    (i == len(self.trace_proj)-(self.nb_body-1)*3):

                pt_size = self.point_size
                #if i == (len(self.trace_proj)-1):
                #    pt_size = self.point_size+2
                    
                pygame.draw.circle(self.screen, col, (int(body[0]),  int(body[1])), pt_size)

            # opengl mode => write tour 2D pygame surface into the texture (which will be be rendered in a quad by the fragment shader)
            if self.use_opengl:
                try:
                    self.frame_tex.write(self.display.get_view('1'))
                    #self.frame_tex.write(self.display.get_buffer())
                except:
                    pass

                self.vao.render(mode=mgl.TRIANGLE_STRIP)

                self.show_options_ui()
                imgui.render()
                self.imgui_renderer.render(imgui.get_draw_data())
                
            # display
            pygame.display.flip()

            # record video
            if self.record_video:
                self.video_recorder.capture_frame(self.display)

            # fps
            self.delta_time = self.clock.tick(self.max_fps)
            self.get_fps()
            self.frames += 1

# -----------------------------------------------------------------------------------------------------------
# https://github.com/bsavery/ray-tracing-one-weekend-taichi/blob/main/main.py
# https://github.com/taichi-dev/taichi_dem/blob/main/dem.py

# python3 main.py --arch=cpu --body=3 --fps=-1
# python3 main.py --arch=gpu --body=10000 --fps=60
# python3 main.py --arch=cpu --body=5 --fps=-1 -rv="h264" -vfps=240
# python3 main.py --arch=cpu --body=3 --fps=-1 -utfm # GPU matrix

def main():

    pygame.init()
    pygame.mouse.set_visible(True)
    pygame.font.init()

    # const
    USE_PROFILER  = 0

    # args
    parser = argparse.ArgumentParser(description="Leapfrog N-Body")

    parser.add_argument('-a', '--arch', help='Taichi backend', default="cpu", action="store")
    parser.add_argument('-f', '--fps', help='Max FPS, -1 for unlimited', default=-1, type=int)
    parser.add_argument('-b', '--body', help='NB Body', default=5, type=int)
    parser.add_argument('-rv', '--record_video', help='', default="", type=str)
    parser.add_argument('-vfps', '--video_fps', help='', default=60, type=int)
    parser.add_argument('-utfm', '--use_taichi_for_matrix', help='', action="store_true")

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
    app = App(screen_width=1280, screen_height=800, use_opengl=1, orbital_mode=1, record_video=args["record_video"], video_fps=args["video_fps"], max_fps=args["fps"], 
              nb_body=args["body"], use_taichi_for_matrix=args["use_taichi_for_matrix"], dt=0.0009, eps=0.05)
    app.run()

if __name__ == "__main__":
    main()