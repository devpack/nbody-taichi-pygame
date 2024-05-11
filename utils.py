import time, collections
import glm
import cv2
import pygame

# -----------------------------------------------------------------------------------------------------------

class Camera:
    
    def __init__(self, app, orbital_mode=False, orbital_speed=0.1, fov=50, near=0.1, far=100, position=(0, 0, 4), speed=0.009, sensivity=0.07, yaw=-90, pitch=0, ortho=False):
        self.app = app

        self.orbital_mode = orbital_mode
        self.orbital_speed = orbital_speed

        self.fov = fov 
        self.near = near 
        self.far = far 
        self.position = glm.vec3(position)
        self.target = glm.vec3(0, 0, 0) # center of the world
        self.aspect_ratio = app.screen_width / app.screen_height

        self.speed = speed
        self.sensivity = sensivity

        self.up = glm.vec3(0, 1, 0)
        self.right = glm.vec3(1, 0, 0)
        self.local_up = glm.vec3(0, 1, 0)

        if self.orbital_mode:
            self.forward = glm.normalize(self.target - self.position)
        else:
            self.forward = glm.vec3(0, 0, -1) # z front

        self.yaw = yaw
        self.pitch = pitch

        self.m_view = self.get_view_matrix()
        self.m_proj = self.get_projection_matrix(ortho=ortho)

    def rotate(self, mouse_dx, mouse_dy):
        self.yaw += mouse_dx * self.sensivity
        self.pitch -= mouse_dy * self.sensivity
        self.pitch = max(-89, min(89, self.pitch))

    def update_camera_vectors(self):

        if self.orbital_mode:
            self.forward = glm.normalize(self.target - self.position)
            self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0, 1, 0)))
            self.up = glm.normalize(glm.cross(self.right, self.forward))

        else:
            yaw, pitch = glm.radians(self.yaw), glm.radians(self.pitch)

            self.forward.x = glm.cos(yaw) * glm.cos(pitch)
            self.forward.y = glm.sin(pitch)
            self.forward.z = glm.sin(yaw) * glm.cos(pitch)

            self.forward = glm.normalize(self.forward)
            self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0, 1, 0)))
            self.up = glm.normalize(glm.cross(self.right, self.forward))

    def update(self, mouse_dx, mouse_dy, forward, backward, left, right, up, down):
        self.move(forward, backward, left, right, up, down)
        self.rotate(mouse_dx, mouse_dy)

        self.update_camera_vectors()
        self.m_view = self.get_view_matrix()

    def move(self, forward, backward, left, right, up, down):

        if self.orbital_mode:
            velocity = self.speed * self.app.delta_time * self.orbital_speed
            self.position -= self.right * velocity

            if forward:
                self.position += self.forward * velocity
            if backward:
                self.position -= self.forward * velocity
            if up:
                self.position -= self.up * velocity
            if down:
                self.position += self.up * velocity

        else:
            velocity = self.speed * self.app.delta_time

            if forward:
                self.position += self.forward * velocity
            if backward:
                self.position -= self.forward * velocity
            if right:
                self.position += self.right * velocity
            if left:
                self.position -= self.right * velocity
            if up:
                self.position -= self.up * velocity
            if down:
                self.position += self.up * velocity

        if self.fov < 1:
            self.fov = 1
        if self.fov > 60:
            self.fov = 60

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.position + self.forward, self.up)
        #return glm.lookAt(self.position, glm.vec3(0), self.up)

    def get_projection_matrix(self, ortho=False):
        if ortho:
            return glm.ortho(-1.0, 1.0, -1.0, 1.0, self.near, self.far)
        else:
            return glm.perspective(glm.radians(self.fov), self.aspect_ratio, self.near, self.far)

# -----------------------------------------------------------------------------------------------------------

class ShaderProgram:

    def __init__(self, ctx):
        self.ctx = ctx
        self.programs = {}
        self.programs['screen'] = self.get_program('screen')

    def get_program(self, shader_name):
        try:
            with open(f'shaders/{shader_name}_vs.glsl') as file:
                vertex_shader = file.read()

            with open(f'shaders/{shader_name}_fs.glsl') as file:
                fragment_shader = file.read()

            program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
            return program
        except Exception as e:
            print("Failed to load %s : %s" % (shader_name, repr(e)))
            return None

    def destroy(self):
        for program in self.programs.values():
            if program:
                program.release()

# -----------------------------------------------------------------------------------------------------------

class FPSCounter:
    def __init__(self):
        self.time = time.perf_counter()
        self.frame_times = collections.deque(maxlen=60)

    def tick(self):
        t1 = time.perf_counter()
        dt = t1 - self.time
        self.time = t1
        self.frame_times.append(dt)

    def get_fps(self):
        total_time = sum(self.frame_times)
        if total_time == 0:
            return 0
        else:
            return len(self.frame_times) / sum(self.frame_times)

# -----------------------------------------------------------------------------------------------------------

class ScreenRecorder:
    def __init__(self, width, height, fps, codec="XVID", out_file='output.avi'):
        print(f'Initializing ScreenRecorder with parameters width:{width} height:{height} fps:{fps}.')
        print(f'Output of the screen recording saved to {out_file}.')

        # define the codec and create a video writer object
        four_cc = cv2.VideoWriter_fourcc(*codec)

        self.video = cv2.VideoWriter(out_file, four_cc, float(fps), (width, height))

    def capture_frame(self, surf):
        # transform the pixels to the format used by open-cv
        pixels = cv2.rotate(pygame.surfarray.pixels3d(surf), cv2.ROTATE_90_CLOCKWISE)
        pixels = cv2.flip(pixels, 1)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

        # write the frame
        self.video.write(pixels)

    def end_recording(self):
        self.video.release()