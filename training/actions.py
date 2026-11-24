"""Eight-coordinate joint actions: relative arm, absolute gripper."""

ACTION_CONTRACT = 'relative_arm_absolute_gripper_v1'


def encode_actions(anchor, targets):
    result = targets.clone() if hasattr(targets, 'clone') else targets.copy()
    result[..., :7] -= anchor[..., None, :7]
    return result


def decode_actions(anchor, actions):
    result = actions.clone() if hasattr(actions, 'clone') else actions.copy()
    result[..., :7] += anchor[..., None, :7]
    return result
