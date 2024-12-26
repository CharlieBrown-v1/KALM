from transformers import LlamaConfig


class LlataConfig(LlamaConfig):
    model_type = "llata"

    def __init__(
        self,
        observation_type="continuous",
        observation_dim=10,
        action_type="discrete",
        action_dim=40,
        action_loss_ratio=0.5,
        max_trajectory_length=50,
        traj_ln=True,
        # do_load_embedding=False,
        # pretrain_dir=None,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=0.000001,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000,
        rope_scaling=None,
        attention_bias=False,
        **kwargs
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            pad_token_id,
            bos_token_id,
            eos_token_id,
            pretraining_tp,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            attention_bias,
            **kwargs
        )

        self.observation_type = observation_type
        self.observation_dim = observation_dim
        self.action_type = action_type
        self.action_dim = action_dim
        self.max_trajectory_length = max_trajectory_length

        # whether use layer norm for trajectory embeddings
        self.traj_ln = traj_ln

        # needed to cheak wheather conflict with LlataForTrajectoryGenerationConfig
        self.action_loss_ratio = action_loss_ratio

        # lky waste: load pretrained observation/action embedding or train new
        # self.do_load_embedding=do_load_embedding
        # self.pretrain_dir=pretrain_dir

        self.traj_start_layer = 0  # lky: add this to set where traj start

class LlataForTrajectoryGenerationConfig(LlataConfig):
    def __init__(
        self,
        observation_type="continuous",
        observation_dim=10,
        action_type="discrete",
        action_dim=40,
        action_loss_ratio=0.5,
        max_trajectory_length=50,
        traj_ln=True,
        # do_load_embedding=False,
        # pretrain_dir=None,
        scale_loss_ratio=0.001,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=0.000001,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000,
        rope_scaling=None,
        attention_bias=False,
        **kwargs
    ):
        super().__init__(
            observation_type,
            observation_dim,
            action_type,
            action_dim,
            max_trajectory_length,
            traj_ln,
            # do_load_embedding,
            # pretrain_dir,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            pad_token_id,
            bos_token_id,
            eos_token_id,
            pretraining_tp,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            attention_bias,
            **kwargs
        )

        self.action_loss_ratio = action_loss_ratio
        # self.pretrain_dir=pretrain_dir
        self.scale_loss_ratio = scale_loss_ratio  # add this to control scale_loss

