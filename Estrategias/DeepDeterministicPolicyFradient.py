import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque

# Configuración inicial
class DDPGTrader:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hiperparámetros
        self.gamma = 0.99
        self.tau = 0.005
        self.learning_rate_actor = 0.001
        self.learning_rate_critic = 0.002
        
        # Redes
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()
        
        # Optimizadores
        self.actor_optimizer = tf.keras.optimizers.Adam(self.learning_rate_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.learning_rate_critic)
        
        # Memoria de experiencias
        self.buffer = deque(maxlen=10000)
        
    def build_actor(self):
        model = tf.keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(self.state_size,)),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='tanh')  # -1 a 1 para acciones
        ])
        return model
    
    def build_critic(self):
        state_input = layers.Input(shape=(self.state_size,))
        action_input = layers.Input(shape=(self.action_size,))
        
        merged = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(256, activation='relu')(merged)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        output = layers.Dense(1, activation='linear')(x)
        
        return tf.keras.Model([state_input, action_input], output)
    
    def get_action(self, state, exploration_noise=0.1):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = self.actor(state)[0]
        
        # Añadir ruido para exploración
        noise = np.random.normal(0, exploration_noise, size=self.action_size)
        action = action.numpy() + noise
        
        # Limitar entre -1 y 1
        return np.clip(action, -1, 1)
    
    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return
        
        # Samplear batch de la memoria
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # Calcular target Q-values
            target_actions = self.target_actor(next_states)
            target_q_values = self.target_critic([next_states, target_actions])
            target_q = rewards + self.gamma * target_q_values * (1 - dones)
            
            # Calcular Q-values actuales
            current_q = self.critic([states, actions])
            
            # Calcular pérdida del crítico
            critic_loss = tf.reduce_mean(tf.square(current_q - target_q))
        
        # Actualizar crítico
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.critic.trainable_variables)
        )
        
        with tf.GradientTape() as tape:
            # Actualizar actor
            actions_pred = self.actor(states)
            critic_value = self.critic([states, actions_pred])
            actor_loss = -tf.reduce_mean(critic_value)
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables)
        )
        
        # Actualizar redes objetivo
        self.update_target_networks()
    
    def update_target_networks(self):
        for target, source in zip(self.target_actor.variables, self.actor.variables):
            target.assign(self.tau * source + (1 - self.tau) * target)
        
        for target, source in zip(self.target_critic.variables, self.critic.variables):
            target.assign(self.tau * source + (1 - self.tau) * target)

def prepare_data(df, features=None):
    """
    Prepara y normaliza los datos de un DataFrame.
    - Si la columna es numérica: aplica estandarización (z-score).
    - Si la columna es categórica: aplica codificación numérica.
    """
    df_normalized = df.copy()
    
    # Si no se pasan features, usar todas
    if features is None:
        features = df.columns.tolist()
    
    for feature in features:
        if feature in df.columns:
            if pd.api.types.is_numeric_dtype(df[feature]):
                # Normalización Z-score
                mean = df[feature].mean()
                std = df[feature].std()
                if std != 0:  # evitar división por cero
                    df_normalized[feature] = (df[feature] - mean) / std
                else:
                    df_normalized[feature] = df[feature] - mean  # solo centrar
            else:
                # Convertir categóricas a números (factorize preserva orden de aparición)
                df_normalized[feature], _ = pd.factorize(df[feature])
    
    return df_normalized

# Función para calcular recompensas
def calculate_reward(current_price, next_price, action, position):
    price_change = (next_price - current_price) / current_price
    
    if action > 0.3:  # Entrada larga
        reward = price_change
    elif action < -0.3:  # Entrada corta
        reward = -price_change
    else:  # Neutral
        reward = -0.001  # Pequeña penalización por no operar
    
    return reward

# Entrenamiento principal
def train_ddpg_trader(df,features, episodes=100):
    # Preparar datos
    df_prepared = prepare_data(df, features)
    state_size = len([col for col in df_prepared.columns if col not in ['timestamp', 'open', 'high', 'low']])
    action_size = 1  # Acción única: -1 (corto) a 1 (largo)
    
    # Inicializar agente
    agent = DDPGTrader(state_size, action_size)
    
    for episode in range(episodes):
        total_reward = 0
        position = 0  # 0: neutral, 1: largo, -1: corto
        
        for i in range(len(df_prepared) - 1):
            # Estado actual
            current_state = df_prepared.iloc[i].drop(['timestamp', 'open', 'high', 'low']).values
            current_price = df_prepared.iloc[i]['close']
            
            # Obtener acción
            action = agent.get_action(current_state, exploration_noise=0.1)
            
            # Siguiente estado
            next_state = df_prepared.iloc[i + 1].drop(['timestamp', 'open', 'high', 'low']).values
            next_price = df_prepared.iloc[i + 1]['close']
            
            # Calcular recompensa
            reward = calculate_reward(current_price, next_price, action[0], position)
            total_reward += reward
            
            # Almacenar experiencia
            done = (i == len(df_prepared) - 2)
            agent.remember(current_state, action, reward, next_state, done)
            
            # Entrenar
            agent.train()
            
            # Actualizar posición (para lógica de recompensa)
            if action[0] > 0.3:
                position = 1
            elif action[0] < -0.3:
                position = -1
            else:
                position = 0
        
        print(f"Episodio {episode + 1}, Recompensa total: {total_reward:.4f}")
    
    return agent

# Función para hacer predicciones
def predict_signals(agent, df):
    df_prepared = prepare_data(df)
    signals = []
    
    for i in range(len(df_prepared)):
        state = df_prepared.iloc[i].drop(['timestamp', 'open', 'high', 'low']).values
        action = agent.get_action(state, exploration_noise=0.0)  # Sin exploración
        
        if action[0] > 0.3:
            signals.append('BUY')
        elif action[0] < -0.3:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    
    return signals
"""
# Ejemplo de uso
if __name__ == "__main__":
    # Suponiendo que df es tu DataFrame con columnas:
    features = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'ema_fast', 'ema_slow', 'stochastic_k', 'stochastic_d']
    
    # Entrenar el modelo
    trained_agent = train_ddpg_trader(df, features, episodes=50)
    
    # Generar señales
    df['signal'] = predict_signals(trained_agent, df)
    
    # Mostrar señales
    print(df[['timestamp', 'close', 'signal']].tail(20))
"""