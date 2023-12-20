def camera_setup(self):
        self.env_2_watch = self.debug_flags['cam_target']

    def camera_follow(self, d_pad):
        if(self.input_delay_counter > self.input_delay):
            if(d_pad[1] == 1):
                self.env_2_watch = int(torch.randint(0, self.num_envs, (1,)))
                self.input_delay_counter = 0
                self.debug_flags['control_en'] = False
            if(d_pad[1] == -1):
                self.env_2_watch = 0
                self.input_delay_counter = 0
                self.debug_flags['control_en'] = False
            if(d_pad[0] == 1):
                self.env_2_watch += 1
                self.input_delay_counter = 0
                self.debug_flags['control_en'] = False
            if(d_pad[0] == -1):
                self.env_2_watch -= 1
                self.input_delay_counter = 0
                self.debug_flags['control_en'] = False
            
        # Wrap Around
        if(self.env_2_watch > self.num_envs-1):
            self.env_2_watch = 0
        if(self.env_2_watch < 0):
            self.env_2_watch = int(self.num_envs-1)

        
        num_per_row = int(np.sqrt(self.num_envs))
        spacing = 2
        row_num = int(self.env_2_watch/num_per_row)*spacing
        col_num = (self.env_2_watch % num_per_row)*spacing        
        env_offset = gymapi.Vec3(col_num, row_num, 0.0)
        cam_offset = torch.tensor([[5.0], [0.0], [0.5]], device=self.device)


        Quat = torch.unsqueeze(self.glider_states[self.env_2_watch, 3:7], 0)
        Quat = torch.unsqueeze(quaternion_invert(self.glider_states[self.env_2_watch, 3:7]), 0)
        Quat = torch.roll(Quat, 1, 1)
        # Quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        cam_offset_rot = quaternion_apply(Quat, torch.permute(cam_offset,(1,0)))
        cam_o = gymapi.Vec3(cam_offset_rot[0,0],cam_offset_rot[0,1],cam_offset_rot[0,2])

        # cam_pos = gymapi.Vec3(self.glider_states[self.env_2_watch,0], self.glider_states[self.env_2_watch,1], self.glider_states[self.env_2_watch,2])
        cam_pos = gymapi.Vec3(self.glider_states[self.env_2_watch,0], self.glider_states[self.env_2_watch,1], 5.0)

        # cam_target = gymapi.Vec3(10, self.glider_states[self.env_2_watch,1], self.glider_states[self.env_2_watch,2])
        cam_target = gymapi.Vec3(self.glider_states[self.env_2_watch,0], self.glider_states[self.env_2_watch,1], self.glider_states[self.env_2_watch,2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos+cam_o+env_offset, cam_target+env_offset)
        # self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)