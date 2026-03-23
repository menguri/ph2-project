from .abstract import ActorCriticBase
from .rnn import ScannedRNN, ActorCriticRNN
from .cnn import ActorCriticCNN
from jaxmarl.environments.overcooked_v2.common import Actions
import jax.numpy as jnp


def get_actor_critic(config) -> ActorCriticBase:
    model_config = dict(config["model"])
    if "ACTION_PREDICTION" in config:
        model_config["ACTION_PREDICTION"] = config["ACTION_PREDICTION"]
    # Forward CycleTransformer (CT) global config keys into model_config so
    # ActorCriticRNN can read them from self.config.
    for _ct_key in (
        "TRANSFORMER_ACTION",
        "TRANSFORMER_WINDOW_SIZE",
        "TRANSFORMER_D_C",
        "TRANSFORMER_N_HEADS",
        "TRANSFORMER_N_LAYERS",
        "TRANSFORMER_RECON_COEF",
        "TRANSFORMER_PRED_COEF",
        "TRANSFORMER_CYCLE_COEF",
        "TRANSFORMER_V2",          # v2 pixel-space recon 아키텍처 분기용
        "TRANSFORMER_STATE_SHAPE", # v2 pixel decoder output shape (H, W, C_full)
    ):
        if _ct_key in config:
            model_config[_ct_key] = config[_ct_key]

    match model_config["TYPE"]:
        case "RNN":
            actor_critic = ActorCriticRNN
        case "CNN":
            actor_critic = ActorCriticCNN
        case _:
            raise NotImplementedError("Only RNN and CNN models are supported.")

    return actor_critic(
        # env.action_space(env.agents[0]).n,
        len(Actions),
        config=model_config,
    )


def initialize_carry(config, batch_size: int):
    model_config = config["model"]

    if model_config["TYPE"] == "RNN":
        gru_state = ActorCriticRNN.initialize_carry(
            batch_size,
            model_config["GRU_HIDDEN_DIM"],
        )
        # When CycleTransformer is enabled, extend the carry with window state.
        if bool(config.get("TRANSFORMER_ACTION", False)):
            W = int(config.get("TRANSFORMER_WINDOW_SIZE", 16))
            D_obs = int(model_config["GRU_HIDDEN_DIM"])
            obs_window = jnp.zeros((batch_size, W, D_obs))
            step_idx = jnp.zeros(batch_size, dtype=jnp.int32)
            return (gru_state, obs_window, step_idx)
        return gru_state

    return None
