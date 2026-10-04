import torch
import typing
if typing.TYPE_CHECKING:
    from resources.envs._base_task import BaseTask, BaseTaskCfg

class BasePolicy:
    def __init__(self, args):
        '''initialize model'''
        self.model = None

    def encode_obs(self, observation):
        '''encode observation'''
        return observation

    def eval(self, task: 'BaseTask', observation):
        obs = self.encode_obs(observation)
        instruction = task.instruction
        
        actions = self.model.get_action(observation)
        
        for action in actions:
            exec_succ, eval_succ = task.take_action(actions, action_type='qpos')
            
            observation = task._get_observations()
            obs = self.encode_obs(observation)
            self.model.update_obs(obs)
 
    def reset(self):
        if self.model is not None:
            self.model.reset()
 
    def close(self):
        if self.model is not None and hasattr(self.model, 'close'):
            self.model.close()
    
    def save(self, img, tag=None):
        from PIL import Image
        from PIL import ImageDraw, ImageFont
        
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        obs = Image.fromarray(img)

        draw = ImageDraw.Draw(obs)
        font = ImageFont.load_default()

        if tag is not None:
            draw.text((obs.width-100, obs.height-60), f'{tag:03d}', fill=(255, 0, 0), font=font)
        obs.save(f'{self.__class__.__name__}_{tag}.png')