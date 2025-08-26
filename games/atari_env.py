import gymnasium as gym
import ale_py
from typing import Optional, Tuple, Any, Dict
from .preprocessor import AtariPreprocessor


class AtariGame:
    """Atari environment wrapper with preprocessing and rendering support."""
    
    def __init__(
        self,
        env_id: str,
        env_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        gym.register_envs(ale_py)
        self.env_id = env_id
        self.env_kwargs = env_kwargs or {}
        self.seed = seed
        self.env = self._make_env(render_mode=None)
        self.render_env = None
        self.render_active = False

    def _make_env(self, render_mode: Optional[str]) -> gym.Env:
        kwargs = dict(self.env_kwargs)
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        env = gym.make(self.env_id, **kwargs)
        
        env = AtariPreprocessor(env, frame_stack=4)
        
        if self.seed is not None:
            try:
                env.reset(seed=self.seed)
            except TypeError:
                pass
        return env

    def ensure_render_env(self) -> None:
        if self.render_env is None:
            self.render_env = self._make_env(render_mode="human")

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        return self.env.reset()

    def step(self, action: int):
        return self.env.step(action)

    def reset_render(self) -> Tuple[Any, Dict[str, Any]]:
        """Reset render environment safely - DON'T close, just reset!"""
        try:
            if self.render_env is not None:
                result = self.render_env.reset()
                self.render_active = True
                return result
            else:
                self.ensure_render_env()
                if self.render_env is not None:
                    result = self.render_env.reset()
                    self.render_active = True
                    return result
                else:
                    return None, {}
        except Exception as e:
            print(f"Render reset error: {e}")
            try:
                if self.render_env is not None:
                    self.render_env.close()
            except:
                pass
            self.render_env = None
            self.render_active = False
            return None, {}

    def step_render(self, action: int):
        """Step render environment safely"""
        if not self.render_active or self.render_env is None:
            return None, 0, True, True, {}
            
        try:
            step_out = self.render_env.step(action)
            self.render_env.render()
            return step_out
        except Exception as e:
            print(f"Render step error: {e}, disabling render")
            self.render_active = False
            return None, 0, True, True, {}

    @property
    def action_space_n(self) -> int:
        return self.env.action_space.n

    def close(self) -> None:
        """Close environments safely"""
        try:
            if self.env is not None:
                self.env.close()
        except Exception as e:
            print(f"Error closing main env: {e}")
            
        try:
            if self.render_env is not None:
                self.render_env.close()
                self.render_env = None
        except Exception as e:
            print(f"Error closing render env: {e}")
            
        self.render_active = False


