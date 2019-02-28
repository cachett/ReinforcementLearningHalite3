#!/usr/bin/env python3
# Python 3.6
import os
import sys
stdout = sys.stderr
sys.stderr = open(os.devnull, 'w')
import numpy as np
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda
from keras.layers.merge import Add
import tensorflow as tf
from math import *
import random
from PrioritizedExperienceReplayBuffer import *
import pickle
import logging
import hlt
from hlt import constants
from hlt.positionals import Direction, Position
from hlt.entity import Dropoff
sys.stderr = stdout


def compute_state(ship, game_map, me):
    """BUT: retourner un vecteur qui représente bien l'état de la partie"""
    sight_window = 17
    array_halite = np.zeros(145)
    array_ship =  np.zeros(145)
    array_dropoff =  np.zeros(145)
    counter = 0

    for i in range(-(sight_window//2), sight_window//2 + 1):
        for j in range(-(sight_window//2) + abs(i), sight_window//2 - abs(i) + 1):
            cell = game_map[Position(i + ship.position.x, j + ship.position.y)]
            array_halite[counter] = cell.halite_amount
            if cell.structure:
                if cell.structure.id == me.shipyard.id or cell.structure.id in me._dropoffs:
                    array_dropoff[counter] = 1
                else:
                    array_dropoff[counter] = -1

            if cell.ship:
                if me.has_ship(cell.ship.id):
                    array_ship[counter] = 1
                else:
                    array_ship[counter] = -1
            counter += 1


    my_dropoffs = list(me._dropoffs.values())
    my_dropoffs.append(me.shipyard)
    distance_my_dropoffs = np.array([game_map.calculate_distance(ship.position, dropoff.position) for dropoff in my_dropoffs])
    min_distance_dropoff = min(distance_my_dropoffs)
    position_closest_dropoff = my_dropoffs[np.argmin(distance_my_dropoffs)].position

    game_state = np.append(np.append(array_halite, array_ship), array_dropoff)
    interesting_values = np.array([len(me._dropoffs) + 1, len(me._ships), ship.position.x, ship.position.y, position_closest_dropoff.x, position_closest_dropoff.y, min_distance_dropoff, ship.halite_amount])
    game_state = np.append(game_state, interesting_values)
    return game_state

def maj_inspired(ship, game_map, me):
    sight_window = 9
    counter = 0
    for i in range(-(sight_window//2), sight_window//2 + 1):
        for j in range(-(sight_window//2) + abs(i), sight_window//2 - abs(i) + 1):
            cell = game_map[Position(i + ship.position.x, j + ship.position.y)]
            if cell.ship and not me.has_ship(cell.ship.id):
                counter += 1
                if counter == 2:
                    return 1
    return 0

def simulation_move(ship, new_position, game_map, me):
    #Déplacement sur la map et calcul du nouveau halite amount et modification de la position du ship
    cell = game_map[ship]
    if cell.ship and cell.ship.id == ship.id:
        cell.ship = None
    ship.halite_amount -= int(0.1*cell.halite_amount)
    ship.position = new_position
    new_cell = game_map[ship]
    #Gestion des cas spéciaux
    if new_cell.has_structure and new_cell.structure.owner == me.id:#dépot du halite au dropoff
        me.halite_amount += ship.halite_amount
        ship.halite_amount = 0

    new_cell.ship = ship
    futur_position.add(ship.position)



def do_action(ship, action, game_map, me):
    if action == 0: #MOVE NORTH
        new_position = game_map.normalize(Position(ship.position.x, ship.position.y - 1))
        simulation_move(ship, new_position, game_map, me)
    elif action == 1: #MOVE SOUTH
        new_position = game_map.normalize(Position(ship.position.x, ship.position.y + 1))
        simulation_move(ship, new_position, game_map, me)
    elif action == 2: #MOVE EAST
        new_position = game_map.normalize(Position(ship.position.x + 1, ship.position.y))
        simulation_move(ship, new_position, game_map, me)
    elif action == 3: #MOVE WEST
        new_position = game_map.normalize(Position(ship.position.x - 1, ship.position.y))
        simulation_move(ship, new_position, game_map, me)
    elif action == 4: #STAY STILL
        cell = game_map[ship]
        gain = int(ceil(0.25*cell.halite_amount))
        mem_halite_amount = ship.halite_amount
        ship.is_inspired = maj_inspired(ship, game_map, me)
        if ship.is_inspired == 1:
            ship.halite_amount = min(1000, ship.halite_amount + 3*gain)
            cell.halite_amount -= (ship.halite_amount - mem_halite_amount)//3
        else:
            ship.halite_amount = min(1000, ship.halite_amount + gain)
            cell.halite_amount -= (ship.halite_amount - mem_halite_amount)
        futur_position.add(ship.position)
    else: #TURN INTO DROPOFF
        cell = game_map[ship]
        new_dropoff =  Dropoff(me.id, constants.OWN_ID, Position(ship.position.x, ship.position.y))
        cell.structure = new_dropoff
        cell.ship = None
        cell.halite_amount = 0
        me._dropoffs[constants.OWN_ID] = new_dropoff
        constants.OWN_ID += 1
        me._ships.pop(ship.id)

def valid_move(ship, game_map):
    #Déplacement sur la map et calcul du nouveau halite amount et modification de la position du ship
    cell = game_map[ship]
    ship_halite_amount = ship.halite_amount - ceil(0.1*cell.halite_amount)
    if ship_halite_amount < 0: #Can't move
        return False
    return True

def valid_stay_still(ship, game_map):
    if game_map[ship].structure or ship.halite_amount > 900:
        return False
    return True

def valid_turn_into_dropoff(ship, game_map, my_halite_amount):
    if (my_halite_amount + ship.halite_amount + game_map[ship].halite_amount) < 4000 or game_map[ship].structure != None:
        return False #Trop cher
    sight_window = 11
    surronding_halite = 0
    for i in range(-(sight_window//2), sight_window//2 + 1):
        for j in range(-(sight_window//2) + abs(i), sight_window//2 - abs(i) + 1):
            surronding_halite += game_map[Position(i + ship.position.x, j + ship.position.y)].halite_amount
    if surronding_halite < 9000:
        return False #Pas assez de halite à coté
    return True

def compute_valid_action(ship, game_map, me):
    valid_movement = valid_move(ship, game_map)
    valid_action = np.array([valid_movement,
                             valid_movement,
                             valid_movement,
                             valid_movement,
                             valid_stay_still(ship, game_map)], dtype=bool)
    return valid_action

def get_new_position(action, ship):
    return game_map.normalize(Position(ship.position.x + directions[a][0], ship.position.y + directions[a][1]))

def get_next_pos_halite_amount(ship, game_map):
    next_pos_halite_amount = np.zeros(5)
    for index, direction in enumerate(directions):
        next_pos_halite_amount[index] = game_map[Position(ship.position.x + direction[0], ship.position.y + direction[1])].halite_amount
    return next_pos_halite_amount


def predict_futur_enemy_pos(my_dropoffs):
    for enemy in enemies:
        for ship in enemy.get_ships():
            if ship.halite_amount > 800: #Bateau très rempli, on s'en fou de la collision
                continue
            if ship.halite_amount < ceil(0.1*game_map[ship.position].halite_amount): # le bateau ne peut pas bouger
                futur_position.add(ship.position)
            elif ship.position in my_dropoffs:
                continue
            else:
                next_pos_halite_amount = get_next_pos_halite_amount(ship, game_map)
                if max(next_pos_halite_amount[:4]) < next_pos_halite_amount[4] / 2 and next_pos_halite_amount[4] > 300: #Je suppose qu'il va rester sur place
                    futur_position.add(ship.position)
                else: # je sais pas je mets tout
                    for direction in directions:
                        futur_position.add(game_map.normalize(Position(ship.position.x + direction[0], ship.position.y + direction[1]))) # Je suppose qu'il bouge à une meilleur case

def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true,y_pred)

def create_model(input_size, action_size, LEARNING_RATE, model_name):
    input = Input(shape=(input_size,))
    shared = Dense(128, activation='relu', kernel_initializer='he_uniform')(input)
    shared = Dense(128, activation='relu', kernel_initializer='he_uniform')(shared)

    # network separate state value and advantages
    advantage_fc = Dense(128, activation='relu', kernel_initializer='he_uniform')(shared)
    advantage = Dense(action_size, activation='linear', kernel_initializer='he_uniform')(advantage_fc)

    value_fc = Dense(128, activation='relu', kernel_initializer='he_uniform')(shared)
    value =  Dense(1)(value_fc)

    # combine the two streams
    advantage = Lambda(lambda advantage: advantage - tf.reduce_mean(advantage, axis=-1, keepdims=True))(advantage)
    value = Lambda(lambda value: tf.tile(value, [1, action_size]))(value)
    q_value = Add()([value, advantage])

    model = Model(inputs=input, outputs=q_value)
    model.compile(loss=huber_loss, optimizer='adam', metrics=['mae'])
    return model

# buffer = Memory(40000)
# pickle.dump(buffer, open("buffer4.data", "wb"))
buffer = pickle.load(open("buffer4.data", "rb"))
loadmodel = True
gamma = 0.95
LEARNING_RATE = 0.001
eps = float(sys.argv[1])
directions = [Direction.North, Direction.South, Direction.East, Direction.West, Direction.Still]
actions = np.array([0,1,2,3,4])

ecart_type = 12
reward_gaussian_weight = np.zeros(128)
for i in range(128):
    reward_gaussian_weight[i] = 3*exp(-i**2/(2*ecart_type**2))


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

if loadmodel:
   model = load_model('my_model.h5', custom_objects={'huber_loss': huber_loss, 'tf': tf})
else:
   model = create_model(443, 5, LEARNING_RATE, 'my_model.h5')

secondDropoff = False
total_halite = 0
total_reward = 0
counter = 0
hash_dim_turn = {32:400, 40:425, 48:450, 56:475, 64:500}
np.random.seed()

# This game object contains the initial game state.
game = hlt.Game()
game.ready("DQN-PER-Para4-GoodMove")
""" <<<Game Loop>>> """
while True:
    game.update_frame()
    me = game.me
    enemies = [game.players[id] for id in game.players.keys() if id != me.id]
    game_map = game.game_map
    my_halite_amount = me.halite_amount
    command_queue = []
    futur_position = set()
    new_state = np.array([])
    state = np.array([])
    my_ships = me.get_ships()
    my_dropoffs = me.get_dropoffs() + [me.shipyard]
    my_dropoffs = [drop.position for drop in my_dropoffs]
    max_turn_number = hash_dim_turn[game_map.width]
    #On ajoute aux futurs positions, tous les ships qui ne vont pas bouger
    for ship in my_ships:
        if ship.halite_amount < ceil(0.1*game_map[ship.position].halite_amount):
            futur_position.add(ship.position)

    #On ajoute les futurs position des adversaire pour éviter les collisions
    predict_futur_enemy_pos(my_dropoffs)

    #On itère sur les ship par ordre de halite amount
    for ship in sorted(my_ships, key=lambda x: -x.halite_amount):
        # Fait un dropoff au piff le 300e tour
        if not secondDropoff and game.turn_number > int(max_turn_number*0.45) and game.turn_number <= int(max_turn_number*0.65) and valid_turn_into_dropoff(ship, game_map, my_halite_amount):
            command_queue.append(ship.make_dropoff())
            mem_cell_halite = game_map[ship].halite_amount
            do_action(ship, 5, game_map, me)
            my_halite_amount -= 4000 - (ship.halite_amount + mem_cell_halite)
            futur_position.discard(ship.position)#l'enlève si il y es seulement
            secondDropoff = True
            continue

                ######################   SELECTION OF ACTION #####################
        valid_action_bool = compute_valid_action(ship, game_map, me)
        valid_action = actions[valid_action_bool]

        if len(valid_action) <= 1: #No choice to do
            command_queue.append(ship.stay_still())
            do_action(ship, 4, game_map, me)
            futur_position.add(ship.position)
        else:
            state = compute_state(ship, game_map, me)
            target_vec = model.predict(state.reshape(1,-1))[0]

            if np.random.random() < eps:#Choix aléatoire
                index = np.random.randint(0, len(valid_action))
                a = valid_action[index]
                #On cherche une position libre
                while get_new_position(a, ship) in futur_position:
                    valid_action = np.delete(valid_action, index)
                    if len(valid_action) == 0: #Toutes les positions potentiels sont occupées, on bouge pas. Collision potentielle
                        a = 4
                        break
                    index = np.random.randint(0, len(valid_action))
                    a = valid_action[index]

            else:#Meilleur choix suivant DNN (modulo les invalid moves)
                target_vec_valid = target_vec[valid_action_bool]
                index = np.argmax(target_vec_valid)
                a = valid_action[index]
                #On cherche une position libre
                while get_new_position(a, ship) in futur_position:
                    target_vec_valid = np.delete(target_vec_valid, index)
                    valid_action = np.delete(valid_action, index)
                    if len(target_vec_valid) == 0: #Toutes les positions potentiels sont occupées, on bouge pas. Collision potentielle
                        a = 4
                        break
                    index = np.argmax(target_vec_valid)
                    a = valid_action[index]

            #On l'ajoute à la command queue
            command_queue.append(ship.move(directions[a]))

            #Simulation de l'action et calcul the new state
            my_hal_before = me.halite_amount
            ship_hal_before = ship.halite_amount
            do_action(ship, a, game_map, me)
            ship_hal_after = ship.halite_amount
            my_hal_after = me.halite_amount
            if a == 4: #stay_still, pas besoin de tout recompute
                new_state = np.copy(state)
                new_state[-1] = ship.halite_amount
                new_state[72] = game_map[ship.position].halite_amount #Indice à changer si on change sight_windows
            else:
                new_state = compute_state(ship, game_map, me)

            #Calcul de la reward
            deplacement = state[-2] - new_state[-2]
            if deplacement < 0:
                punition = 50
            else:
                punition = 0
            gain_halite = my_hal_after - my_hal_before
            reward = gain_halite * 100 + (deplacement * ship.halite_amount * reward_gaussian_weight[int(new_state[-2])]) + (ship_hal_after - ship_hal_before) - punition

            #On supprime les prochaines actions infaisable pour la Qvalue
            target_vec_prime = model.predict(new_state.reshape(1,-1))[0]
            if not valid_move(ship, game_map): #Il ne reste que stay still
                target = reward + gamma * target_vec_prime[-1]
            elif not valid_stay_still(ship,game_map): #On peut bouger mais pas rester sur place dans l'état suivant
                target = reward + gamma * np.max(target_vec_prime[:4])
            else:#On peut tout faire dans l'état suivant
                target = reward + gamma * np.max(target_vec_prime)

            #Sauve l'experience dans le buffer
            td_error = abs(target_vec[a] - target)
            target_vec[a] = target
            buffer.add(td_error, [state, target_vec])


    if game.turn_number < int(max_turn_number*0.45) and my_halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(game.me.shipyard.spawn())

    if game.turn_number == max_turn_number:
        pickle.dump(buffer, open("buffer4.data", "wb"))
    game.end_turn(command_queue)
