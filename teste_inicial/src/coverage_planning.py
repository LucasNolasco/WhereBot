#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid, GridCells
from map_msgs.msg import OccupancyGridUpdate
from std_msgs.msg import Bool
import geometry_msgs
from geometry_msgs.msg import PointStamped, Point, Polygon
import tf
import numpy as np
from math import sqrt, atan2, sin, cos
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib import SimpleGoalState
from threading import Lock
import cv2
from random import randrange

# Dimensão do robô: 13,8 cm x 17,8 cm

AREA_THRESHOLD = 1500 # Área mínima para que o robô visite

GRIDS_DISTANCE_PATH_BUILDING = 12 # Em pixels (TODO: Verificar se não é melhor usar em metros)
ROBOT_INFLUENCE = 5 # Região ao redor do robô para ser considerada como visitada (Área total da região visitada: (2 * ROBOT_INFLUENCE + 1) x (2 * ROBOT_INFLUENCE + 1))

DEBUG_SHOW_IMAGES = False # Flag para indicar se as imagens de debug devem ser exibidas

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

            #rospy.loginfo("Costmap atualizado")

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
    #   calcReachablePoints
    #       Gera um mapa com todos os pontos alcançáveis para o robô.
    #       Para isso, utiliza a estratégia de encontrar uma região
    #       contínua com valores no costmap dentro de uma faixa que
    #       indica a não colisão.       
    #
    #   Parâmetros: Coordenadas atuais do robô
    #
    #   Retorno: Área total que o robô pode alcançar, 
    #            tupla com os limites da área alcançável
    #
    # ----------------------------------------------------------
    def calcReachablePoints(self, x, y):
        stack = []
        checked = np.zeros((self.costmap.shape[0], self.costmap.shape[1]))

        for i in range(x - ROBOT_INFLUENCE, x + ROBOT_INFLUENCE + 1):
            for j in range(y - ROBOT_INFLUENCE, y + ROBOT_INFLUENCE + 1):
                stack.append((i,j))
                checked[j][i] = 1

        min_x, min_y, max_x, max_y = x, y, x, y              

        self.reachable_map = np.zeros((self.costmap.shape[0], self.costmap.shape[1]))

        area = 0
        while len(stack) > 0: # Implementa um flood fill, marcando todos os pontos alcançáveis à partir da posição inicial do robô
            (x, y) = stack.pop()

            if x > 0 and x < self.costmap.shape[1] - 1 and y > 0 and y < self.costmap.shape[0] - 1:
                area += 1
                self.reachable_map[y][x] = 1

                if checked[y][x - 1] == 0 and self.isFree(x - 1, y):
                    stack.append((x - 1, y))
                    checked[y][x - 1] = 1
                if checked[y][x + 1] == 0 and self.isFree(x + 1, y):
                    stack.append((x + 1, y))
                    checked[y][x + 1] = 1
                if checked[y - 1][x] == 0 and self.isFree(x, y - 1):
                    stack.append((x, y - 1))
                    checked[y - 1][x] = 1
                if checked[y + 1][x] == 0 and self.isFree(x, y + 1):
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

        return area, (min_x, min_y, max_x, max_y)

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
        MARGIN = 5

        total_area, (x0, y0, xn, yn) = self.calcReachablePoints(x, y) # Calcula a matriz de pontos alcançáveis e também a quantidade total de pontos

        calcLimitsRate = rospy.Rate(10)
        while x0 < MARGIN or y0 < MARGIN or yn > self.costmap.shape[0] - MARGIN or xn > self.costmap.shape[1] - MARGIN or total_area < 100: # Caso as margens encontradas para o ambiente não tenham sido encontradas
            rospy.loginfo("Failed to stablish environment limits. Retrying")
            calcLimitsRate.sleep()
            total_area, (x0, y0, xn, yn) = self.calcReachablePoints(x, y) # Calcula a matriz de pontos alcançáveis e também a quantidade total de pontos

        checked = np.zeros(self.reachable_map.shape) # Matriz para marcar os pontos já checados
        stack = []
        current_cost = None
        field = []

        # Percorre toda a matriz procurando os pontos marcados como alcançáveis 
        for ry in range(self.reachable_map.shape[0]):
            for rx in range(self.reachable_map.shape[1]):
                # Pontos alcançáveis que ainda não foram visitados ou checados são adicionados na pilha
                if self.reachable_map[ry][rx] == 1 and checked[ry][rx] == 0 and self.visited_points[ry][rx] == 0: 
                    stack.append((rx, ry))
                    checked[ry][rx] = 1

                    area = 0
                    area_map = np.zeros(self.costmap.shape)
                    area_y = np.array([])

                    furthest_x = rx # Variáveis para armazenar os pontos da região mais distantes do robô
                    furthest_y = ry

                    field_points = []

                    while len(stack) > 0: # Loop para avaliar todos os pontos empilhados
                        (px, py) = stack.pop()
                        if px >= 0 and px < self.reachable_map.shape[1] and py >= 0 and py < self.reachable_map.shape[0] and self.visited_points[py][px] == 0: # Verifica se é uma coordenada válida
                            area += 1        
                            area_map[py][px] = 1

                            field_points.append((px, py))

                            if self.distance((furthest_x, furthest_y), (x, y)) < self.distance((px, py), (x, y)): # Verifica se esse novo ponto é o mais distante
                                furthest_x = px
                                furthest_y = py

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

                        px = furthest_x
                        py = furthest_y

                        # Calcula o custo para o objetivo dividindo a área total pela distância (TODO: Custo não é um bom nome, tenho que trocar isso)
                        if px != x or py != y:
                            cost = area / self.distance((px, py), (x, y))
                        else:
                            cost = 0

                        if current_cost is None or cost > current_cost:
                            field = np.copy(field_points) # Polígono com a área ainda inexplorada
                            
                            current_goal_cost = cost 
                            rospy.loginfo("Region of Interest Area: {0}".format(area))

                            if DEBUG_SHOW_IMAGES == True: # Exibe para debug a área que está sendo analisada
                                cv2.imshow("AREA MAP", area_map)
                                cv2.waitKey(1)

        # Calcula os limites da região escolhida para explorar
        region_x0, region_xn, region_y0, region_yn = self.costmap.shape[1], 0, self.costmap.shape[0], 0
        for point in field:
            if point[0] < region_x0:
                region_x0 = point[0]
            if point[0] > region_xn:
                region_xn = point[0]
            if point[1] < region_y0:
                region_y0 = point[1]
            if point[1] > region_yn:
                region_yn = point[1]

        rospy.loginfo("Environment limits: ({0}, {1}), ({2}, {3})".format(region_x0, region_y0, region_xn, region_yn))

        y_axis = np.arange(region_y0, region_yn, GRIDS_DISTANCE_PATH_BUILDING) # Gera um eixo com os pontos de interesse dentro da região a ser explorada
        x_axis = np.arange(region_x0, region_xn, GRIDS_DISTANCE_PATH_BUILDING) 

        path = [] # Adiciona todos do pontos alcançáveis do grid como objetivos
        for yi in y_axis:
            for xi in x_axis:
                if self.reachable_map[yi][xi] == 1:
                    path.append((xi, yi))

        final_path = [] # Caminho final reordenado
        removal = -1 # Id do objetivo já reordenado que deve ser removido da lista antiga
        final_path.append((x,y,0)) # Adiciona a posição inicial do robô
        for point in final_path: # Reordena os objetivos para criar um caminho escolhendo o próximo ponto sempre como o mais próximo ao anterior
            next_point = None
            for i, available_point in enumerate(path):
                if next_point is None or self.distance(next_point, final_path[-1]) > self.distance(available_point, final_path[-1]):
                    theta = atan2((available_point[1] - final_path[-1][1]), (available_point[0] - final_path[-1][0]))

                    next_point = (available_point[0], available_point[1], theta)
                    removal = i

            if next_point is not None:
                final_path.append(next_point)
                del path[removal]

        final_path.pop(0) # Remove a posição inicial do robô

        #path.sort(key=(lambda xi: (sqrt((xi[0] - x)**2 + (xi[1] - y)**2)))) # Ordenando pela distância com relação à posição inicial do robô. Não é mais utilizado
        
        if len(final_path) > 0:
            rospy.loginfo("New path calculated, {0} steps".format(len(final_path)))
        else:
            rospy.loginfo("No path found")

        return final_path, total_area

    # ----------------------------------------------------------
    #   distance
    #       Calcula a distância euclidiana entre dois pontos
    #
    #   Parâmetros:
    #       point1 - Tupla no formato (x, y)
    #       point2 - Tupla no formato (x, y)
    #   Retorno: Coordenadas x e y do robô dentro do mapa
    #
    # ----------------------------------------------------------
    def distance(self, point1, point2):
        if point1 is None or point2 is None:
            return -1

        return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

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

    # ----------------------------------------------------------
    #   publishCoveredArea
    #       Publica os dados de pontos explorados pelo robô no
    #       tópico covered_space para a visualização no RViZ
    #
    # ----------------------------------------------------------
    def publishCoveredArea(self):
        covered_msg = GridCells()
                
        covered_msg.header.frame_id = "map"

        covered_msg.cell_width = self.scale
        covered_msg.cell_height = self.scale

        for i in range(self.costmap.shape[0]):
            for j in range(self.costmap.shape[1]):
                if self.visited_points[i][j] == 1:
                    point = Point()
                    point.x = (j + 0.5) * self.scale + self.origin_x
                    point.y = (i + 0.5) * self.scale + self.origin_y
                    point.z = 0
                    covered_msg.cells.append(point)

        self.covered_space.publish(covered_msg)

    # ----------------------------------------------------------
    #   setVisitedPoints
    #       Marca a região visitada com base na posição do robô
    #
    #   Parametros: 
    #       x: Coordenada x do robô dentro do mapa
    #       y: Coordenada y do robô dentro do mapa
    #       coverage_dim: Área ao redor da posição do robô a ser
    #                     marcada como visitada           
    #
    # ----------------------------------------------------------
    def setVisitedPoints(self, x, y, coverage_dim = ROBOT_INFLUENCE):
        for i in range(y - coverage_dim, y + coverage_dim + 1): # Marca uma área coberta pela movimentação do robô
            for j in range(x - coverage_dim, x + coverage_dim + 1):
                if i < self.visited_points.shape[0] and j < self.visited_points.shape[1]:
                    self.visited_points[i][j] = 1

    # ----------------------------------------------------------
    #   isFree
    #       Verifica se uma determinada posição possui obstáculo
    #
    #   Parâmetros:
    #       x: Coordenada do robô no eixo x
    #       y: Coordenada do robô no eixo y
    #
    #   Retorno: True, caso não haja obstáculo naquela posição
    #            False, caso haja obstáculo naquela posição
    #
    # ----------------------------------------------------------
    def isFree(self, x, y):
        if self.costmap[y][x] >= 0 and self.costmap[y][x] < 100:
            return True
        else:
            return False

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

        self.covered_space = rospy.Publisher("/covered_space", GridCells, queue_size=10) # Tópico responsável por publicar a área coberta pela movimentação do robô

        client = actionlib.SimpleActionClient('move_base',MoveBaseAction) # Cria o cliente do move_base para enviar os objetivos para onde o robô deve ir
        rospy.loginfo("Waiting for move_base server")
        client.wait_for_server() # Espera a move_base estar configurada
        rospy.loginfo("Connection established")

        self.tf_listener = tf.TransformListener() # Instancia o objeto para lidar com as transformações de posição no mapa

        self.path = None

        no_goal = 0
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            try:
                x, y = self.getPos() # Obtém a posição do robô no mapa

                if self.visited_points is not None:
                    self.publishCoveredArea() # Publica o caminho percorrido pela navegação para ser exibido no RViZ

                    rospy.loginfo("Visited: ({0}, {1})".format(x, y))

                    covered_area = np.sum(np.sum(self.visited_points,axis=0),axis=0) # Área total percorrida pela navegação
                    rospy.loginfo("Approx. Covered Area: {0}".format(covered_area))

                    if DEBUG_SHOW_IMAGES == True: # Exibe algumas imagens de debug mostrando o costmap binarizado e o caminho percorrido pelo robô
                        plot_bin_costmap = np.zeros(self.costmap.shape) # Cria um costmap binarizado
                        for a in range(plot_bin_costmap.shape[0]):
                            for b in range(plot_bin_costmap.shape[1]):
                                if self.isFree(b, a):
                                    plot_bin_costmap[a][b] = 1

                        cv2.imshow("Teste", plot_bin_costmap) # Mostra na tela o costmap binarizado
                        cv2.waitKey(1)

                        cv2.imshow("Path", self.visited_points) # Mostra na tela o caminho percorrido pelo robô
                        cv2.waitKey(1)
                    
                    if self.map_finished:
                        self.setVisitedPoints(x, y) # Marca a região visitada

                    if self.map_finished and client.simple_state == SimpleGoalState.DONE:
                        if self.path is None or len(self.path) == 0:
                            self.path, area = self.findNextGoal(x, y) # Calcula o próximo objetivo do robô

                        new_goal = None

                        if self.path is not None and len(self.path) > 0:
                            new_goal = self.path.pop(0) # Pega o próximo objetivo

                            # Caso o objetivo já tenha sido visitado ou haja algum obstáculo, pega o próximo
                            while (self.visited_points[new_goal[1]][new_goal[0]] or not self.isFree(new_goal[0], new_goal[1])) and len(self.path) > 0:
                                new_goal = self.path.pop(0)

                            #new_goal = (new_goal[0], new_goal[1])

                        rospy.loginfo("Moving to goal: {0}".format(new_goal))
                        
                        if new_goal is not None: # Caso tenha sido encontrado um novo objetivo
                            no_goal = 0

                            goal = MoveBaseGoal()
                            goal.target_pose.header.frame_id = "map" 
                            goal.target_pose.header.stamp = rospy.Time.now()
                            goal.target_pose.pose.position.x = (new_goal[0] + 0.5) * self.scale + self.origin_x
                            goal.target_pose.pose.position.y = (new_goal[1] + 0.5) * self.scale + self.origin_y
                            goal.target_pose.pose.orientation.w = cos(new_goal[2]/2)
                            goal.target_pose.pose.orientation.z = sin(new_goal[2]/2)

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
