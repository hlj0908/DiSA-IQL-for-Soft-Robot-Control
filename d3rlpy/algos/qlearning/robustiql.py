import dataclasses
from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import ActionSpace
from ...models.builders import (
    create_continuous_q_function,
    create_normal_policy,
    create_value_function,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.q_functions import MeanQFunctionFactory
from ...optimizers.optimizers import OptimizerFactory, make_optimizer_field
from ...types import Shape
from .base import QLearningAlgoBase
# 🔹 使用新的 RobustIQLImpl
from .torch.robust_iql_impl import RobustIQLImpl
from .torch.iql_impl import IQLModules


__all__ = ["RobustIQLConfig", "RobustIQL"]


@dataclasses.dataclass()
class RobustIQLConfig(LearnableConfig):
    r"""Robust Implicit Q-Learning algorithm.

    修改自 IQL，在 Bellman backup 中引入 distributionally robust penalty。
    """

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    value_encoder_factory: EncoderFactory = make_encoder_field()
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    n_critics: int = 2
    expectile: float = 0.7
    weight_temp: float = 3.0
    max_weight: float = 100.0

    # 🔹 robust 参数
    use_robust: bool = True
    robust_alpha: float = 0.1
    uncertainty_set: str = "Wasserstein"   # 可选 ["Wasserstein", "KL", "Chi2", "TV"]
    decay_schedule: bool = True            # penalty 是否随训练衰减

    def create(self, device: DeviceArg = False, enable_ddp: bool = False) -> "RobustIQL":
        return RobustIQL(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "robust_iql"


class RobustIQL(QLearningAlgoBase[RobustIQLImpl, RobustIQLConfig]):
    def inner_create_impl(self, observation_shape: Shape, action_size: int) -> None:
        policy = create_normal_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            min_logstd=-5.0,
            max_logstd=2.0,
            use_std_parameter=True,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        q_funcs, q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_q_funcs, targ_q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        value_func = create_value_function(
            observation_shape,
            self._config.value_encoder_factory,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )

        actor_optim = self._config.actor_optim_factory.create(
            policy.named_modules(),
            lr=self._config.actor_learning_rate,
            compiled=self.compiled,
        )
        critic_optim = self._config.critic_optim_factory.create(
            list(q_funcs.named_modules()) + list(value_func.named_modules()),
            lr=self._config.critic_learning_rate,
            compiled=self.compiled,
        )

        modules = IQLModules(
            policy=policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            value_func=value_func,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
        )

        # 🔹 使用 RobustIQLImpl
        self._impl = RobustIQLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            expectile=self._config.expectile,
            weight_temp=self._config.weight_temp,
            max_weight=self._config.max_weight,
            compiled=self.compiled,
            device=self._device,
            use_robust=self._config.use_robust,
            robust_alpha=self._config.robust_alpha,
            uncertainty_set=self._config.uncertainty_set,
            decay_schedule=self._config.decay_schedule,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


# 注册成可调用算法
register_learnable(RobustIQLConfig)


