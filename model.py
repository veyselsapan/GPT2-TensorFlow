"""
Contains the definition of the GPT-2 model. 
"""
import tensorflow as tf
from config import Config
import einops
import math
from tf.keras.activations import gelu


class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = self.add_weight(name="LayerNorm_W", shape=(cfg.d_model,), initializer="ones", trainable=True)
        self.b = self.add_weight(name="LayerNorm_b", shape=(cfg.d_model,), initializer="zeros", trainable=True)

    def call(self, residual):
        # residual: [batch, position, d_model]
        if self.cfg.debug:
            print("Residual:", residual)

        mean = einops.reduce(residual, "bpd->bp1", "mean")
        residual = residual - mean

        variance = einops.reduce(tf.square(residual), "bpd->bp1", "mean")
        scale = tf.sqrt(variance + self.cfg.layer_norm_eps)
        normalized = residual / scale

        normalized = normalized * self.w + self.b

        if self.cfg.debug:
            print("Normalized:", residual)

        return normalized
    
    def get_config(self):
        return {"d_model": self.cfg.d_model,
                "debug": self.cfg.debug,
                "layer_norm_eps": self.cfg.layer_norm_eps,
                "d_vocab": self.cfg.d_vocab,
                "init_range": self.cfg.init_range,
                "block_size": self.cfg.block_size,
                "d_head": self.cfg.d_head,
                "d_mlp": self.cfg.d_mlp,
                "n_heads":self.cfg.n_heads,
                "n_layers": self.cfg.n_layers}
    
    @classmethod
    def from_config(cls, config):
        # Create a new instance of the class using the config
        return cls(Config(**config))

class Embed(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = self.add_weight(name="Embed_W_E", 
            shape=(cfg.d_vocab, cfg.d_model),
            initializer=tf.random_normal_initializer(stddev=cfg.init_range),
            trainable=True
        )

    def call(self, tokens):
        # tokens: [batch, position]
        if self.cfg.debug:
            print("Tokens:", tokens)

        # TensorFlow's equivalent for PyTorch's advanced indexing
        tokens = tf.cast(tokens, dtype=tf.int32)
        embed = tf.gather(self.W_E, tokens, axis=0) # [batch, position, d_model]

        if self.cfg.debug:
            print("Embeddings:", embed)

        return embed
    
    def get_config(self):
        return {"d_model": self.cfg.d_model,
                "debug": self.cfg.debug,
                "layer_norm_eps": self.cfg.layer_norm_eps,
                "d_vocab": self.cfg.d_vocab,
                "init_range": self.cfg.init_range,
                "block_size": self.cfg.block_size,
                "d_head": self.cfg.d_head,
                "d_mlp": self.cfg.d_mlp,
                "n_heads":self.cfg.n_heads,
                "n_layers": self.cfg.n_layers}
    
    @classmethod
    def from_config(cls, config):
        # Create a new instance of the class using the config
        return cls(Config(**config))
    
class PosEmbed(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = self.add_weight(name='PosEmbed_W_pos',
            shape=(cfg.block_size, cfg.d_model),
            initializer=tf.random_normal_initializer(stddev=cfg.init_range),
            trainable=True
        )

    def call(self, tokens):
        # tokens: [batch, position]
        if self.cfg.debug:
            print("Tokens:", tokens)
        tokens = tf.cast(tokens, dtype=tf.int32)
        pos_embed = self.W_pos[:tf.shape(tokens)[1], :] # [position, d_model]
        pos_embed = einops.repeat(pos_embed, "position d_model -> batch position d_model", batch=tf.shape(tokens)[0])

        if self.cfg.debug:
            print("Pos_embed:", pos_embed)

        return pos_embed

    def get_config(self):
        return {"d_model": self.cfg.d_model,
                "debug": self.cfg.debug,
                "layer_norm_eps": self.cfg.layer_norm_eps,
                "d_vocab": self.cfg.d_vocab,
                "init_range": self.cfg.init_range,
                "block_size": self.cfg.block_size,
                "d_head": self.cfg.d_head,
                "d_mlp": self.cfg.d_mlp,
                "n_heads":self.cfg.n_heads,
                "n_layers": self.cfg.n_layers}
    
    @classmethod
    def from_config(cls, config):
        # Create a new instance of the class using the config
        return cls(Config(**config))

class Attention(tf.keras.layers.Layer):
    def __init__(self, cfg, dropout_rate=0.1):
        super().__init__()
        self.cfg = cfg
        self.W_Q = self.add_weight(name="Attention_W_Q", shape=(cfg.n_heads, cfg.d_model, cfg.d_head), initializer="random_normal", trainable=True)
        self.b_Q = self.add_weight(name="Attention_b_Q", shape=(cfg.n_heads, cfg.d_head), initializer="zeros", trainable=True)
        self.W_K = self.add_weight(name="Attention_W_K", shape=(cfg.n_heads, cfg.d_model, cfg.d_head), initializer="random_normal", trainable=True)
        self.b_K = self.add_weight(name="Attention_b_K", shape=(cfg.n_heads, cfg.d_head), initializer="zeros", trainable=True)
        self.W_V = self.add_weight(name="Attention_W_V", shape=(cfg.n_heads, cfg.d_model, cfg.d_head), initializer="random_normal", trainable=True)
        self.b_V = self.add_weight(name="Attention_b_V", shape=(cfg.n_heads, cfg.d_head), initializer="zeros", trainable=True)
        self.W_O = self.add_weight(name="Attention_W_O", shape=(cfg.n_heads, cfg.d_head, cfg.d_model), initializer="random_normal", trainable=True)
        self.b_O = self.add_weight(name="Attention_b_O", shape=(cfg.d_model,), initializer="zeros", trainable=True)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.IGNORE = -1e5

    def call(self, normalized_resid_pre, training=False):
        # normalized_resid_pre: [batch, position, d_model]
        if self.cfg.debug:
            print("Normalized_resid_pre:", normalized_resid_pre)
            print("W_Q:", self.W_Q)

        q = tf.einsum("bqd,hdm->bqhm", normalized_resid_pre, self.W_Q) + self.b_Q
        k = tf.einsum("bkd,hdm->bkhm", normalized_resid_pre, self.W_K) + self.b_K

        attn_scores = tf.einsum("bqhd,bkhd->bhqk", q, k)
        attn_scores /= math.sqrt(self.cfg.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)

        pattern = tf.nn.softmax(attn_scores, axis=-1)
        if training:
            pattern = self.dropout(pattern, training=training)

        v = tf.einsum("bkd,hdm->bkhm", normalized_resid_pre, self.W_V) + self.b_V
        z = tf.einsum("bhqk,bkhd->bqhd", pattern, v)

        attn_out = tf.einsum("bqhd,hdm->bqm", z, self.W_O) + self.b_O
        return attn_out

    def apply_causal_mask(self, attn_scores):
        mask = 1 - tf.linalg.band_part(tf.ones((tf.shape(attn_scores)[-2], tf.shape(attn_scores)[-1])), -1, 0)
        attn_scores = tf.where(mask == 1, self.IGNORE, attn_scores)
        return attn_scores

    def get_config(self):
        return {"d_model": self.cfg.d_model,
                "debug": self.cfg.debug,
                "layer_norm_eps": self.cfg.layer_norm_eps,
                "d_vocab": self.cfg.d_vocab,
                "init_range": self.cfg.init_range,
                "block_size": self.cfg.block_size,
                "d_head": self.cfg.d_head,
                "d_mlp": self.cfg.d_mlp,
                "n_heads":self.cfg.n_heads,
                "n_layers": self.cfg.n_layers}
    
    @classmethod
    def from_config(cls, config):
        # Create a new instance of the class using the config
        return cls(Config(**config))

class MLP(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = self.add_weight(name="MLP_W_in",
            shape=(cfg.d_model, cfg.d_mlp),
            initializer=tf.random_normal_initializer(stddev=cfg.init_range),
            trainable=True
        )
        self.b_in = self.add_weight(name="MLP_b_in",
            shape=(cfg.d_mlp,),
            initializer="zeros",
            trainable=True
        )
        self.W_out = self.add_weight(name="MLP_W_out",
            shape=(cfg.d_mlp, cfg.d_model),
            initializer=tf.random_normal_initializer(stddev=cfg.init_range),
            trainable=True
        )
        self.b_out = self.add_weight(name="MLP_b_out",
            shape=(cfg.d_model,),
            initializer="zeros",
            trainable=True
        )

    def call(self, normalized_resid_mid):
        # normalized_resid_mid: [batch, position, d_model]
        if self.cfg.debug:
            print("Normalized_resid_mid:", normalized_resid_mid)

        pre = tf.einsum("bpd, dm -> bpm", normalized_resid_mid, self.W_in) + self.b_in
        post = gelu(pre)
        mlp_out = tf.einsum("bpm, md -> bpd", post, self.W_out) + self.b_out

        return mlp_out

    def get_config(self):
        return {"d_model": self.cfg.d_model,
                "debug": self.cfg.debug,
                "layer_norm_eps": self.cfg.layer_norm_eps,
                "d_vocab": self.cfg.d_vocab,
                "init_range": self.cfg.init_range,
                "block_size": self.cfg.block_size,
                "d_head": self.cfg.d_head,
                "d_mlp": self.cfg.d_mlp,
                "n_heads":self.cfg.n_heads,
                "n_layers": self.cfg.n_layers}
    
    @classmethod
    def from_config(cls, config):
        # Create a new instance of the class using the config
        return cls(Config(**config))

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, cfg, dropout_rate=0.1):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, resid_pre):
        # resid_pre [batch, position, d_model]
        normalized_resid_pre = self.ln1(resid_pre)
        attn_out = self.attn(normalized_resid_pre, training=training)
        attn_out = self.dropout1(attn_out, training=training)
        resid_mid = resid_pre + attn_out
        normalized_resid_mid = self.ln2(resid_mid)
        mlp_out = self.mlp(normalized_resid_mid)
        mlp_out = self.dropout2(mlp_out, training=training)
        resid_post = resid_mid + mlp_out
        return resid_post

    def get_config(self):
        return {"d_model": self.cfg.d_model,
                "debug": self.cfg.debug,
                "layer_norm_eps": self.cfg.layer_norm_eps,
                "d_vocab": self.cfg.d_vocab,
                "init_range": self.cfg.init_range,
                "block_size": self.cfg.block_size,
                "d_head": self.cfg.d_head,
                "d_mlp": self.cfg.d_mlp,
                "n_heads":self.cfg.n_heads,
                "n_layers": self.cfg.n_layers}
    
    @classmethod
    def from_config(cls, config):
        # Create a new instance of the class using the config
        return cls(Config(**config))
    
class Unembed(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = self.add_weight(name="Unembed_W_U",
            shape=(cfg.d_model, cfg.d_vocab),
            initializer=tf.random_normal_initializer(stddev=cfg.init_range),
            trainable=True
        )
        self.b_U = self.add_weight(name="Unembed_b_U",
            shape=(cfg.d_vocab,),
            initializer="zeros",
            trainable=False
        )

    def call(self, normalized_resid_final):
        # normalized_resid_final [batch, position, d_model]
        if self.cfg.debug:
            tf.print("Normalized_resid_final:", normalized_resid_final)

        logits = tf.einsum("bpd, dv -> bpv", normalized_resid_final, self.W_U) + self.b_U
        return logits

    def get_config(self):
        return {"d_model": self.cfg.d_model,
                "debug": self.cfg.debug,
                "layer_norm_eps": self.cfg.layer_norm_eps,
                "d_vocab": self.cfg.d_vocab,
                "init_range": self.cfg.init_range,
                "block_size": self.cfg.block_size,
                "d_head": self.cfg.d_head,
                "d_mlp": self.cfg.d_mlp,
                "n_heads":self.cfg.n_heads,
                "n_layers": self.cfg.n_layers}
    
    @classmethod
    def from_config(cls, config):
        # Create a new instance of the class using the config
        return cls(Config(**config))
    
class GPT(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def call(self, tokens, training=False):
        # tokens [batch, position]
        embed = self.embed(tokens)
        pos_embed = self.pos_embed(tokens)
        residual = embed + pos_embed
        for block in self.blocks:
            residual = block(residual, training=training)

        normalized_resid_final = self.ln_final(residual)
        logits = self.unembed(normalized_resid_final)
        # logits have shape [batch, position, d_vocab]
        if self.cfg.debug:
            print("tokens:", tokens)
            print("logits:", logits)
        return logits

    def get_config(self):
        return {"d_model": self.cfg.d_model,
                "debug": self.cfg.debug,
                "layer_norm_eps": self.cfg.layer_norm_eps,
                "d_vocab": self.cfg.d_vocab,
                "init_range": self.cfg.init_range,
                "block_size": self.cfg.block_size,
                "d_head": self.cfg.d_head,
                "d_mlp": self.cfg.d_mlp,
                "n_heads":self.cfg.n_heads,
                "n_layers": self.cfg.n_layers}
    
    @classmethod
    def from_config(cls, config):
        # Create a new instance of the class using the config
        return cls(Config(**config))

    @tf.function(reduce_retracing=True)
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if tf.shape(idx)[1] <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits = self.call(idx_cond, training=False)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_k_value = tf.minimum(top_k, tf.shape(logits)[-1])
                values, _ = tf.math.top_k(logits, k=top_k_value)
                min_values = values[:, -1, tf.newaxis]
                logits = tf.where(logits < min_values, -float('Inf'), logits)

            probs = tf.nn.softmax(logits, axis=-1)
            idx_next = tf.random.categorical(probs, num_samples=1, dtype=tf.int32)
            idx = tf.concat([idx, idx_next], axis=1)

        return idx
