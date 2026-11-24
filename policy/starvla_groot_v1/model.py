"""GR00T with state+action queries and no learnable future tokens."""
from training.models import StarVLA

class StarVLAGrootV1(StarVLA):
    def __init__(self,cfg,dim):
        super().__init__(dict(cfg,head_action_queries_only=False,action_future_tokens=0),dim)
