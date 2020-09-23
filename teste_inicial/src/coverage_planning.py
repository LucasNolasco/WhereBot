#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from std_msgs.msg import Bool
import geometry_msgs
from geometry_msgs.msg import PointStamped
import tf
import numpy as np
from math import sqrt
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib import SimpleGoalState
from threading import Lock
import cv2
from random import randrange

ROBOT_PADDING = 0.2 # Dimensão do robô: 13,8 cm x 17,8 cm (Arredondei para 20 x 20)
AREA_THRESHOLD = 1500 # Área mínima para que o robô visite

DEBUG_SHOW_IMAGES = True # Flag para indicar se as imagens de debug devem ser exibidas

class CoveragePlanning:
    # ----------------------------------------------------------
    #   createCostTranslationTable
    #       Calcula um vetor com o valor das conversões para os valores
    #       do costmap
    #
    # ----------------------------------------------------------
    def createCostTranslationTable(self):
        self.cost_translation_table = np.zeros((256,))

        for i in range(256):
            self.cost_translation_table[i] = (1 + (251 * (i - 1)) / 97)

        self.cost_translation_table[0] = 0
        self.cost_translation_table[99] = 253
        self.cost_translation_table[100] = 254
    

    # ----------------------------------------------------------
    #   getConvertedCost
    #       Função para converter os valores recebidos pelo tópico do
    #       costmap que variam de -1 até 100 para o padrão de 0 até 255.
    #
    #   Parâmetros: Custo entre -1 e 100
    #
    #   Retorno: Custo convertido no intervalo entre 0 e 255.
    #
    # ----------------------------------------------------------
    def getConvertedCost(self, cost):
        if self.cost_translation_table is None:
            self.createCostTranslationTable()

        if cost == -1:
            return 255
        else:
            return self.cost_translation_table[cost]

    # ----------------------------------------------------------
    #   costmap_callback 
    #       Callback para receber a base do costmap.
    #
    #   Parâmetros: Mensagem do formato nav_msgs::GridOccupancy
    #
    # ----------------------------------------------------------
    def costmap_callback(self, data):
        self.scale = data.info.resolution
        self.origin_x = data.info.origin.position.x
        self.origin_y = data.info.origin.position.y

        if self.costmap is None:
            self.mutex.acquire()
            self.costmap = np.zeros((data.info.height, data.info.width))
            self.visited_points = np.zeros((data.info.height, data.info.width))

            for row in range(data.info.height):
                for column in range(data.info.width):
                    if column < self.costmap.shape[1] and row < self.costmap.shape[0]:
                        self.costmap[row][column] = self.getConvertedCost(data.data[row * data.info.width + column])

            self.mutex.release()

        if data.info.height > self.costmap.shape[0] or data.info.width > self.costmap.shape[1]:
            rospy.loginfo("Dimensões incorretas recebidas. Atual: ({0}, {1}), Recebido: ({2}, {3})".format(self.costmap.shape[1], self.costmap.shape[0], data.info.width, data.info.height))

        rospy.loginfo("Costmap recebido")
    
    # ----------------------------------------------------------
    #   costmap_update_callback 
    #       Callback para receber as atualizações realizadas no
    #       costmap.
    #
    #   Parâmetros: Mensagem do formato nav_msgs::GridOccupancyUpdate
    #
    # ----------------------------------------------------------
    def costmap_update_callback(self, data):
        if self.costmap is not None:
            x0 = data.x
            y0 = data.y

            width = data.width
            height = data.height
            
            if x0 + width > self.costmap.shape[1] or y0 + height > self.costmap.shape[0]:
                rospy.loginfo("Atualização maior do que o mapa, somente uma parte será atualizada")

            self.mutex.acquire()
            for row in range(height):
                for column in range(width):
                    if row < self.costmap.shape[0] and column < self.costmap.shape[1]:
                        self.costmap[row + y0][column + x0] = self.getConvertedCost(data.data[row * width + column])
            self.mutex.release()

            rospy.loginfo("Costmap atualizado")

        else:
            rospy.loginfo("Mapa inexistente. Atualização recebida e descartada")

    # ----------------------------------------------------------
    #   explore_notifications_callback 
    #       Callback para receber as mensagens enviadas pelo tópico
    #       do nó explore que envia a notificação indicando o fim da
    #       exploração para a construção do mapa.
    #
    #   Parâmetros: Mensagem do formato std_msgs::Bool
    #
    # ----------------------------------------------------------
    def explore_notification_callback(self, data):
        self.map_finished = data.data
        if self.map_finished:
            rospy.loginfo("Mapa finalizado. Movimentação iniciando.")

    # ----------------------------------------------------------
    #   oldFindNextGoal
    #       Função antiga que combinava a findReachablePoints e findNextGoal.
    #       No entanto, possui implementação mais confusa, logo foi substituída.
    #       Acredito que não voltaremos a utilizá-la.
    #
    #   Parâmetros: Coordenadas atuais do robô
    #
    #   Retorno: Coordenadas para onde o robô deve se mover
    #
    # ----------------------------------------------------------
    def oldFindNextGoal(self, x, y):
        checked = np.zeros((self.visited_points.shape[0], self.visited_points.shape[1]))
        goal_checked = np.zeros((self.costmap.shape[0], self.costmap.shape[1]))

        stack = []
        stack.append((x, y))
        checked[y][x] = 1
        
        total_area = 0

        goal = None
        current_goal_area = 0
        self.mutex.acquire()
        while len(stack) > 0:
            point = stack.pop(0)
            total_area += 1
            
            if point[0] >= self.costmap.shape[0] - 1 or point[1] >= self.costmap.shape[1] - 1 or point[0] <= 0 or point[1] <= 0:
                continue

            if self.costmap[point[1]][point[0] - 1] >= 1 and self.costmap[point[1]][point[0] - 1] < 128:
                if checked[point[1]][point[0] - 1] == 0:
                    stack.append((point[0] - 1, point[1]))
                    checked[point[1]][point[0] - 1] = 1
            
            if self.costmap[point[1]][point[0] + 1] >= 1 and self.costmap[point[1]][point[0] + 1] < 128:
                if checked[point[1]][point[0] + 1] == 0:
                    stack.append((point[0] + 1, point[1]))
                    checked[point[1]][point[0] + 1] = 1

            if self.costmap[point[1] - 1][point[0]] >= 1 and self.costmap[point[1] - 1][point[0]] < 128:
                if checked[point[1] - 1][point[0]] == 0:
                    stack.append((point[0], point[1] - 1))
                    checked[point[1] - 1][point[0]] = 1

            if self.costmap[point[1] + 1][point[0]] >= 1 and self.costmap[point[1] + 1][point[0]] < 128:
                if checked[point[1] + 1][point[0]] == 0:
                    stack.append((point[0], point[1] + 1))
                    checked[point[1] + 1][point[0]] = 1

            if self.visited_points[point[1]][point[0]] == 0:
                not_visited_stack = []
                not_visited_stack.append(point)
               
                area_map = np.zeros(self.costmap.shape)
                checked[point[1]][point[0]] = 1

                area = 0
                while len(not_visited_stack) > 0 and goal is None:
                    visit_point = not_visited_stack.pop(0)
                    area += 1
                    
                    if visit_point[0] >= self.costmap.shape[1] - 1 or visit_point[1] >= self.costmap.shape[0] - 1 or visit_point[0] <= 0 or visit_point[1] <= 0:
                        continue

                    area_map[visit_point[1]][visit_point[0]] = 1

                    if self.costmap[visit_point[1]][visit_point[0] - 1] >= 1 and self.costmap[visit_point[1]][visit_point[0] - 1] < 128:
                        if self.visited_points[visit_point[1]][visit_point[0] - 1] == 0 and goal_checked[visit_point[1]][visit_point[0] - 1] == 0:
                            not_visited_stack.append((visit_point[0] - 1, visit_point[1]))
                            goal_checked[visit_point[1]][visit_point[0] - 1] = 1

                    if self.costmap[visit_point[1]][visit_point[0] + 1] >= 1 and self.costmap[visit_point[1]][visit_point[0] + 1] < 128:
                        if self.visited_points[visit_point[1]][visit_point[0] + 1] == 0 and goal_checked[visit_point[1]][visit_point[0] + 1] == 0:
                            not_visited_stack.append((visit_point[0] + 1, visit_point[1]))
                            goal_checked[visit_point[1]][visit_point[0] + 1] = 1

                    if self.costmap[visit_point[1] - 1][visit_point[0]] >= 1 and self.costmap[visit_point[1] - 1][visit_point[0]] < 128:
                        if self.visited_points[visit_point[1] - 1][visit_point[0]] == 0 and goal_checked[visit_point[1] - 1][visit_point[0]] == 0:
                            not_visited_stack.append((visit_point[0], visit_point[1] - 1))
                            goal_checked[visit_point[1] - 1][visit_point[0]] = 1

                    if self.costmap[visit_point[1] + 1][visit_point[0]] >= 1 and self.costmap[visit_point[1] + 1][visit_point[0]] < 128:
                        if self.visited_points[visit_point[1] + 1][visit_point[0]] == 0 and goal_checked[visit_point[1] + 1][visit_point[0]] == 0:
                            not_visited_stack.append((visit_point[0], visit_point[1] + 1))
                            goal_checked[visit_point[1] + 1][visit_point[0]] = 1

                if area > AREA_THRESHOLD:
                    if area > current_goal_area:
                        current_goal_area = area
                        goal = visit_point
                        rospy.loginfo("AREA: {0}".format(area))

                        cv2.imshow("AREA MAP", area_map)
                        cv2.waitKey(0)


        self.mutex.release()

        rospy.loginfo("TOTAL AREA: {0}".format(total_area))

        return goal

    # ----------------------------------------------------------
    #   calcReachablePoints
    #       Gera um mapa com todos os pontos alcançáveis para o robô.
    #       Para isso, utiliza a estratégia de encontrar uma região
    #       contínua com valores no costmap dentro de uma faixa que
    #       indica a não colisão.       
    #
    #   Parâmetros: Coordenadas atuais do robô
    #
    #   Retorno: Área total que o robô pode alcançar
    #
    # ----------------------------------------------------------

    def calcReachablePoints(self, x, y):
        stack = []
        stack.append((x,y))

        min_x, min_y, max_x, max_y = x, y, x, y

        checked = np.zeros((self.costmap.shape[0], self.costmap.shape[1]))
        checked[y][x] = 1

        self.reachable_map = np.zeros((self.costmap.shape[0], self.costmap.shape[1]))

        area = 0
        while len(stack) > 0:
            (x, y) = stack.pop()
            #rospy.loginfo("x: {0}, y: {1}, qtd: {2}, cost: {3}".format(x, y, len(stack), self.costmap[y][x]))
            
            if x >= 0 and x < self.costmap.shape[1] and y >= 0 and y < self.costmap.shape[0]:
                area += 1
                self.reachable_map[y][x] = 1

                if checked[y][x - 1] == 0 and self.costmap[y][x - 1] >= 1 and self.costmap[y][x - 1] < 128:
                    stack.append((x - 1, y))
                    checked[y][x - 1] = 1
                if checked[y][x + 1] == 0 and self.costmap[y][x + 1] >= 1 and self.costmap[y][x + 1] < 128:
                    stack.append((x + 1, y))
                    checked[y][x + 1] = 1
                if checked[y - 1][x] == 0 and self.costmap[y - 1][x] >= 1 and self.costmap[y - 1][x] < 128:
                    stack.append((x, y - 1))
                    checked[y - 1][x] = 1
                if checked[y + 1][x] == 0 and self.costmap[y + 1][x] >= 1 and self.costmap[y + 1][x] < 128:
                    stack.append((x, y + 1))
                    checked[y + 1][x] = 1

                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

        rospy.loginfo("Area: {0}, ({1}, {2}), ({3}, {4})".format(area, min_x, min_y, max_x, max_y))

        return area

    # ----------------------------------------------------------
    #   findNextGoal
    #       Calcula o próximo objetivo para onde o robô deve se mover
    #       utilizando o mapa com todas as posições que o robô pode
    #       alcançar dentro do mapa.
    #
    #   Parâmetros: Coordenadas atuais do robô
    #
    #   Retorno: Coordenadas para onde o robô deve se mover e área total
    #            de pontos alcançáveis dentro do mapa
    #
    # ----------------------------------------------------------
    def findNextGoal(self, x, y):
        total_area = self.calcReachablePoints(x, y) # Calcula a matriz de pontos alcançáveis e também a quantidade total de pontos

        checked = np.zeros(self.reachable_map.shape) # Matriz para marcar os pontos já checados
        stack = []
        goal = None
        current_goal_area = 0

        # Percorre toda a matriz procurando os pontos marcados como alcançáveis 
        for ry in range(self.reachable_map.shape[0]):
            for rx in range(self.reachable_map.shape[1]):
                # Pontos alcançáveis que ainda não foram visitados ou checados são adicionados na pilha
                if self.reachable_map[ry][rx] == 1 and checked[ry][rx] == 0 and self.visited_points[ry][rx] == 0: 
                    stack.append((rx, ry))
                    checked[ry][rx] = 1

                    area = 0
                    area_map = np.zeros(self.costmap.shape)
                    area_x = np.array([]) # Vetores para calcular o centroide da região (A princípio não está mais sendo usado)
                    area_y = np.array([])

                    temp_x = rx # Variáveis para armazenar os pontos da região mais distantes do robô
                    temp_y = ry

                    while len(stack) > 0: # Loop para avaliar todos os pontos empilhados
                        (px, py) = stack.pop()
                        if px >= 0 and px < self.reachable_map.shape[1] and py >= 0 and py < self.reachable_map.shape[0] and self.visited_points[py][px] == 0: # Verifica se é uma coordenada válida
                            area += 1        
                            area_map[py][px] = 1
                            area_x = np.append(area_x, px)
                            area_y = np.append(area_y, py)

                            if pow(temp_x - x, 2) + pow(temp_y - y, 2) < pow(px - x, 2) + pow(py - y, 2): # Verifica se esse novo ponto é o mais distante
                                temp_x = px
                                temp_y = py

                            if checked[py][px - 1] == 0 and self.reachable_map[py][px - 1] == 1: # Adiciona os pontos adjacentes na pilha
                                stack.append((px - 1, py))
                                checked[py][px - 1] = 1
                            if checked[py][px + 1] == 0 and self.reachable_map[py][px + 1] == 1:
                                stack.append((px + 1, py))
                                checked[py][px + 1] = 1
                            if checked[py - 1][px] == 0 and self.reachable_map[py - 1][px] == 1:
                                stack.append((px, py - 1))
                                checked[py - 1][px] = 1
                            if checked[py + 1][px] == 0 and self.reachable_map[py + 1][px] == 1:
                                stack.append((px, py + 1))
                                checked[py + 1][px] = 1
 
                    if area > AREA_THRESHOLD: # Verifica se ao final a área é maior que o valor mínimo para ser um objetivo válido
                        #px = int(np.sum(area_x) / area_x.size) # Era usado no cálculo do centróide
                        #py = int(np.sum(area_y) / area_y.size)

                        px = temp_x
                        py = temp_y

                        # Calcula o custo para o objetivo dividindo a área total pela distância (TODO: Custo não é um bom nome, tenho que trocar isso)
                        if px != x or py != y:
                            cost = area / sqrt(pow(px - x, 2) + pow(py - y, 2))
                        else:
                            cost = 0

                        if goal is None or (goal[0] == x and goal[1] == y):
                            current_cost = 0
                        else:
                            current_cost = current_goal_area / sqrt(pow(goal[0] - x, 2) + pow(goal[1] - y, 2))

                        if cost > current_cost:
                            current_goal_area = area
                            
                            if x < px: # Tenta afastar um pouco mais as coordenadas de objetivo das bordas do mapa, evitando colisões
                                px -= 5
                            else:
                                px += 5

                            if y < py:
                                py -= 5
                            else:
                                py += 5

                            goal = (px, py)

                            rospy.loginfo("AREA: {0}".format(area))

                            if DEBUG_SHOW_IMAGES == True: # Exibe para debug a área que está sendo analisada
                                cv2.imshow("AREA MAP", area_map)
                                cv2.waitKey(1)

        return goal, total_area

    # ----------------------------------------------------------
    #   getPos
    #       Calcula a posição do robô dentro das coordenadas do mapa
    #
    #   Retorno: Coordenadas x e y do robô dentro do mapa
    #
    # ----------------------------------------------------------
    def getPos(self):
        x, y = 0, 0
        if self.tf_listener is not None:
            stamped_point = PointStamped()
            stamped_point.header.frame_id = "base_link"
            stamped_point.header.stamp = rospy.Time(0)
            stamped_point.point.x = 0
            stamped_point.point.y = 0
            stamped_point.point.z = 0

            p = self.tf_listener.transformPoint("map", stamped_point)

            x = int((p.point.x - self.origin_x) / self.scale)
            y = int((p.point.y - self.origin_y) / self.scale)

        return x, y

    def __init__(self):
        self.costmap = None # Mapa inflado para evitar que o robô se aproxime de pontos que não é capaz de alcançar
        self.cost_translation_table = None # Vetor com a transformação dos valores recebidos pelo tópico do costmap
        self.visited_points = None # Mapa com os pontos já visitados pelo robô
        self.map_finished = False # Flag que indica se a exploração para a criação do mapa já foi finalizada

        self.reachable_map = None # Mapa com as posições que o robô é capaz de alcançar

        self.mutex = Lock() # Mutex para não atualizar o costmap enquanto o próximo objetivo está sendo calculado

        self.tf_listener = None # Referência para o objeto para acessar o tópico tf que é o responsável pelas transformações

        rospy.init_node('coverage_planning', anonymous=False) # Cria o nó
        rospy.Subscriber("move_base/global_costmap/costmap", OccupancyGrid, self.costmap_callback) # Se inscreve no tópico do costmap
        rospy.Subscriber('move_base/global_costmap/costmap_updates', OccupancyGridUpdate, self.costmap_update_callback) # Se inscreve no tópico para os updates do costmap
        rospy.Subscriber('/explore/explorer_reached_goal', Bool, self.explore_notification_callback) # Se inscreve para o tópico que envia as notificações de fim de exploração

        client = actionlib.SimpleActionClient('move_base',MoveBaseAction) # Cria o cliente do move_base para enviar os objetivos para onde o robô deve ir
        rospy.loginfo("Waiting for move_base server")
        client.wait_for_server() # Espera a move_base estar configurada
        rospy.loginfo("Connection established")

        self.tf_listener = tf.TransformListener() # Instancia o objeto para lidar com as transformações de posição no mapa

        no_goal = 0

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            try:
                x, y = self.getPos() # Obtém a posição do robô no mapa

                if self.visited_points is not None:
                    #robot_dims = int((ROBOT_PADDING / self.scale) / 2)
                    #robot_dims += (robot_dims) % 2

                    robot_dims = 5
                    for i in range(y - robot_dims, y + robot_dims + 1): # Marca uma área coberta pela movimentação do robô
                        for j in range(x - robot_dims, x + robot_dims + 1):
                            if i < self.visited_points.shape[0] and j < self.visited_points.shape[1]:
                                self.visited_points[i][j] = 1

                    rospy.loginfo("Visited: ({0}, {1})".format(x, y))

                    covered_area = np.sum(np.sum(self.visited_points,axis=0),axis=0)
                    rospy.loginfo("Approx. Covered Area: {0}".format(covered_area))

                    if DEBUG_SHOW_IMAGES == True: # Exibe algumas imagens de debug mostrando o costmap binarizado e o caminho percorrido pelo robô
                        my_costmap = np.zeros(self.costmap.shape) # Cria um costmap binarizado
                        for a in range(my_costmap.shape[0]):
                            for b in range(my_costmap.shape[1]):
                                if self.costmap[a][b] >= 1 and self.costmap[a][b] < 128:
                                    my_costmap[a][b] = 1

                        cv2.imshow("Teste", my_costmap)
                        cv2.waitKey(1)

                        cv2.imshow("Path", self.visited_points)
                        cv2.waitKey(1)

                    if self.map_finished and client.simple_state == SimpleGoalState.DONE:
                        new_goal, area = self.findNextGoal(x, y) # Calcula o próximo objetivo do robô

                        rospy.loginfo("Moving to goal: {0}".format(new_goal))
                        
                        if new_goal is not None: # Caso tenha sido encontrado um novo objetivo
                            no_goal = 0

                            goal = MoveBaseGoal()
                            goal.target_pose.header.frame_id = "map" 
                            goal.target_pose.header.stamp = rospy.Time.now()
                            goal.target_pose.pose.position.x = (new_goal[0] + 0.5) * self.scale + self.origin_x
                            goal.target_pose.pose.position.y = (new_goal[1] + 0.5) * self.scale + self.origin_y
                            goal.target_pose.pose.orientation.w = 1.0

                            client.send_goal(goal)
                            rospy.loginfo("New goal sent to move_base")

                        elif new_goal is None and area > AREA_THRESHOLD:
                            no_goal += 1
                        elif new_goal is None and area < AREA_THRESHOLD: # Caso o robô esteja em um ponto cego do mapa pela binarização do costmap
                            rospy.loginfo("Robot on a costmap blindspot. Moving to a random location")

                            goal = MoveBaseGoal()
                            goal.target_pose.header.frame_id = "map" 
                            goal.target_pose.header.stamp = rospy.Time.now()
                            goal.target_pose.pose.position.x = (randrange(0, self.costmap.shape[1]) + 0.5) * self.scale + self.origin_x
                            goal.target_pose.pose.position.y = (randrange(0, self.costmap.shape[0]) + 0.5) * self.scale + self.origin_y
                            goal.target_pose.pose.orientation.w = 1.0

                            client.send_goal(goal)
                            rospy.loginfo("New goal sent to move_base")


                        if no_goal >= 10: # Se a tentativa de encontrar um novo objetivo falhou por 10 vezes seguidas
                            rospy.loginfo("Movimentação finalizada")
                            break

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            
            rate.sleep()

        rospy.spin()

if __name__ == '__main__':
    CoveragePlanning()
