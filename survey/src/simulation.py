#!/usr/bin/python3

import rospy, tf, csv, os
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PointStamped
import numpy as np
from math import sqrt, log10
from threading import Lock
import random


APS_CORDINATES = [(140,240), (75,144)]

CSV_HEADER = ["Grid Point","Drawing X","Drawing Y","2e:20","14:a1"]

MHz = 2417
FSPL = 27.55 #Free-Space Path Loss adapted avarage constant for home WiFI routers and following units


class WirelessSimulation:
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
        self.explore_finished = data.data
        if self.explore_finished:
            self.setVariablesCSV()
            rospy.loginfo("Mapa finalizado. Movimentação iniciando.")

    def movement_finished_callback(self, data):
        string = String()
        self.explore_finished = False
        self.dbm_csv.close()
        os.system('rosrun map_server map_saver -f /home/wagner/catkin_ws/src/survey/simulation_files/my_map')
        string.data = '/home/wagner/catkin_ws/src/survey/simulation_files/my_map.pgm' + ':' +  self.dbm_csv_path
        self.pub_output.publish(string)
        rospy.loginfo("Movimentação finalizada. Salvando mapa e CSV.")

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
        x = int(x)
        y = int(y)
        if self.costmap[y][x] >= 0 and self.costmap[y][x] < 100:
            return True
        else:
            return False

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

    def pixelToMeters(self, positionTuple):
        return ( ((positionTuple[0] + 0.5 )*self.scale + self.origin_x), 
                 ((positionTuple[1] + 0.5 )*self.scale + self.origin_y))

    def countBlackPixels(self, robotPosition, apPosition):
        x0 = robotPosition[0]
        y0 = robotPosition[1]
        x1 = apPosition[0]
        y1 = apPosition[1]
        
        mat = self.costmap

        # Swap axes if Y slope is smaller than X slope
        transpose = abs(x1 - x0) < abs(y1 - y0)
        if transpose:
            mat = mat.T
            x0, y0, x1, y1 = y0, x0, y1, x1
        # Swap line direction to go left-to-right if necessary
        if x0 > x1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        

        # Compute intermediate coordinates using line equation
        x = np.arange(x0 + 1, x1)
        y = np.round(((y1 - y0) / (x1 - x0)) * (x - x0) + y0).astype(x.dtype)
        
        countPixels = 0
        for i in range(len(x)):
            if mat[y[i]][x[i]] > 250:
                countPixels += 1

        return countPixels
        
    def dbmCalculator(self, robotPosition, apPosition):
        blackPixels = self.countBlackPixels(robotPosition, apPosition)
        distance = self.distance(self.pixelToMeters(robotPosition), self.pixelToMeters(apPosition))

        dbm = (-int(20*log10(distance) - FSPL + 20*log10(MHz)))
        
        if blackPixels < 18:
            dbm += (blackPixels/3)*(-2)
        else:
            dbm += -15

        return int(dbm)

    def setVariablesCSV(self):
        self.dbm_csv_path = '/home/wagner/catkin_ws/src/survey/simulation_files/' + str(random.randint(1,100000)) + '.csv'
        self.dbm_csv = open(self.dbm_csv_path, mode='w') # Cria e abre o arquivo onde ficarao salvos os valores de dbm
        self.dbm_writer = csv.writer(self.dbm_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) # Configura o formato do csv
        self.dbm_writer.writerow(CSV_HEADER) # Adiciona no csv o nosso HEADER pre definido

    def __init__(self):
        self.costmap = None # Mapa inflado para evitar que o robô se aproxime de pontos que não é capaz de alcançar
        self.cost_translation_table = None # Vetor com a transformação dos valores recebidos pelo tópico do costmap
        self.explore_finished = False # Flag que indica se a exploração para a criação do mapa já foi finalizada

        self.mutex = Lock() # Mutex para não atualizar o costmap enquanto o próximo objetivo está sendo calculado

        self.tf_listener = None # Referência para o objeto para acessar o tópico tf que é o responsável pelas transformações

        rospy.init_node('coverage_planning', anonymous=False) # Cria o nó
        rospy.Subscriber("move_base/global_costmap/costmap", OccupancyGrid, self.costmap_callback) # Se inscreve no tópico do costmap
        rospy.Subscriber('move_base/global_costmap/costmap_updates', OccupancyGridUpdate, self.costmap_update_callback) # Se inscreve no tópico para os updates do costmap
        rospy.Subscriber('/explore/explorer_reached_goal', Bool, self.explore_notification_callback) # Se inscreve para o tópico que envia as notificações de fim de exploração
        rospy.Subscriber('/movement/finished', String, self.movement_finished_callback) # Se inscreve para o tópico que envia as notificações quando o movimento completa todo

        self.pub_output = rospy.Publisher("/survey/heatmap/input", String, queue_size=10)

        self.tf_listener = tf.TransformListener() # Instancia o objeto para lidar com as transformações de posição no mapa
        
        self.setVariablesCSV()

        rate = rospy.Rate(0.5)
        counter = 1
        while not rospy.is_shutdown():
            if self.explore_finished:
                try:
                    x, y = self.getPos() # Obtém a posição do robô no mapa
                    csv_line = [str(counter), str(x), str(y)]
                    for ap in APS_CORDINATES:
                        dbm = self.dbmCalculator((x,y), ap)
                        csv_line.append(dbm)

                    #rospy.loginfo("CSV Line: ({0})".format(csv_line))
                    self.dbm_writer.writerow(csv_line)
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue
                counter += 1
            rate.sleep()
        rospy.spin()

        self.dbm_csv.close()

if __name__ == '__main__':
    WirelessSimulation()
