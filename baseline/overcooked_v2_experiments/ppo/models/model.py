from .abstract import ActorCriticBase
from .rnn import ScannedRNN, ActorCriticRNN
from .cnn import ActorCriticCNN
from jaxmarl.environments.overcooked_v2.common import Actions


def get_actor_critic(config) -> ActorCriticBase:
    model_config = config["model"]

    match model_config["TYPE"]:
        case "RNN":
            actor_critic = ActorCriticRNN
        case "CNN":
            actor_critic = ActorCriticCNN
        case _:
            raise NotImplementedError("Only RNN and CNN models are supported.")

    # config에 ACTION_DIM이 저장되어 있으면 사용, 없으면 OvercookedV2 기본값(6)
    action_dim = model_config.get("ACTION_DIM", len(Actions))
    return actor_critic(
        action_dim,
        config=model_config,
    )


def initialize_carry(config, batch_size: int):
    model_config = config["model"]

    if model_config["TYPE"] == "RNN":
        return ActorCriticRNN.initialize_carry(batch_size, model_config["GRU_HIDDEN_DIM"])

    return None
