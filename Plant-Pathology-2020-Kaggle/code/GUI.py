import os
import sys

from PyQt5.QtWidgets import (
	QWidget, QPushButton, QApplication,
	QHBoxLayout, QVBoxLayout, QGridLayout,
	QListView, QListWidget, QListWidgetItem,
	QLabel, QMessageBox, QSizePolicy,
	QGraphicsBlurEffect
)
from PyQt5.QtCore import Qt, QStringListModel, QSize, QModelIndex, pyqtSignal
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QPixmap, QIcon, QFont
from PyQt5.Qt import QThread

PREV_SIZE = 480
CATEGORIES = ['healthy', 'multiple_diseases', 'rust', 'scab']

class HardWorkThread(QThread):
	signal = pyqtSignal(dict, str)
	def __init__(self, hard_function, *args, **kwargs):
		super().__init__()
		self.hard_function = hard_function
		self.args = (args)
		self.kwargs = (kwargs)

	def __del__(self):
		self.wait()

	def run(self):
		try:
			raw_y_pred, raw_true_label = self.hard_function(*self.args, **self.kwargs)
			y_pred = {CATEGORIES[i]:float(raw_y_pred[0][i]) for i in range(len(raw_y_pred[0]))}
			true_label = None
			for i in range(len(raw_true_label)):
				if raw_true_label[i]:
					true_label = str(CATEGORIES[i])
		except:
			y_pred, true_label = None, None
		finally:
			self.signal.emit(y_pred, true_label)


class GUIWindow(QWidget):

	def __init__(self, model_predict_function, model_param_path="../output/lala[cpu].pkl", images_path="./images", save_path="./images_output"):
		super().__init__()
		self.model_predict_function = model_predict_function
		self.model_param_path = model_param_path
		self.images_path = images_path
		self.save_path = save_path

		self.file_name = None
		self.predicting = False
	# 	self.init_UI()
	#
	# def init_UI(self):
	# 	self.setGeometry(300, 300, 400, 300)
		self.setWindowTitle('Plant Pathology')
		self.setStyleSheet("background-color: #ffffff")
		# self.setGraphicsEffect(QGraphicsBlurEffect())

		self.hbox1 = QHBoxLayout()

		self.file_list = QListWidget()
		self.file_list.setFixedWidth(150)
		self.file_list.setResizeMode(QListView.Fixed)
		self.file_list.setIconSize(QSize(40, 40))
		self.file_list.setFont(QFont("Arial", 9))
		self.file_list.currentItemChanged.connect(self.on_click_file_list)

		self.qList = self.get_file_list()

		# self.string_list_model = QStringListModel()
		# self.file_list_model = QStandardItemModel()
		for file_name in self.qList:
			item = QListWidgetItem(QIcon(self.getPath(file_name)), file_name)
			self.file_list.addItem(item)
		# self.string_list_model.setStringList(self.qList)
		# self.file_list.setModel(self.file_list_model)


		self.hbox1.addWidget(self.file_list)

		self.vbox1 = QVBoxLayout()

		self.hbox_pictures = QHBoxLayout()

		self.preview_layout = QVBoxLayout()

		# self.prev_stretch_layout = QVBoxLayout()

		self.preview_label = QLabel()
		self.preview_label.setAlignment(Qt.AlignCenter)
		self.preview_label.setFixedSize(PREV_SIZE, PREV_SIZE)
		self.preview_label.setStyleSheet("border: 4px solid #00bbff")
		self.preview_label.setScaledContents(True)
		# preview_label.setPixmap()

		# self.prev_stretch_layout.addWidget(self.preview_label)
		# self.prev_stretch_layout.setStretchFactor(self.preview_label, 1)

		self.preview_notation = QLabel()
		self.preview_notation.setFont(QFont("Arial", 10))
		self.preview_notation.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
		# self.preview_notation.setAlignment(Qt.AlignCenter | Qt.AlignTop)

		self.preview_layout.addWidget(self.preview_label,0,Qt.AlignCenter)
		self.preview_layout.addWidget(self.preview_notation,0,Qt.AlignCenter | Qt.AlignTop)

		self.hbox_pictures.addLayout(self.preview_layout)

		self.vbox_predictions = QVBoxLayout()

		self.raw_attn_layout = QVBoxLayout()

		self.raw_attn_img_label = QLabel()
		self.raw_attn_img_label.setFont(QFont("Arial", 14, QFont.Bold))
		self.raw_attn_img_label.setAlignment(Qt.AlignCenter)
		self.raw_attn_img_label.setFixedSize(PREV_SIZE / 2, PREV_SIZE / 2)
		self.raw_attn_img_label.setStyleSheet("border: 4px solid #cccccc;"
											"color: #aaaaaa")

		self.raw_attn_notation = QLabel("raw attention map")
		self.raw_attn_notation.setFont(QFont("Arial", 10))
		self.raw_attn_notation.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

		self.raw_attn_layout.addWidget(self.raw_attn_img_label,0,Qt.AlignCenter)
		self.raw_attn_layout.addWidget(self.raw_attn_notation,0,Qt.AlignHCenter | Qt.AlignTop)

		self.heat_attn_layout = QVBoxLayout()

		self.heat_attn_img_label = QLabel()
		self.heat_attn_img_label.setFont(QFont("Arial", 14, QFont.Bold))
		self.heat_attn_img_label.setAlignment(Qt.AlignCenter)
		self.heat_attn_img_label.setFixedSize(PREV_SIZE / 2, PREV_SIZE / 2)
		self.heat_attn_img_label.setStyleSheet("border: 4px solid #ffaa00;"
											   "color: #aaaaaa")

		self.heat_attn_notation = QLabel("attention heatmap")
		self.heat_attn_notation.setFont(QFont("Arial", 10))
		self.heat_attn_notation.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

		self.heat_attn_layout.addWidget(self.heat_attn_img_label,0,Qt.AlignCenter)
		self.heat_attn_layout.addWidget(self.heat_attn_notation,0,Qt.AlignHCenter | Qt.AlignTop)

		self.vbox_predictions.addLayout(self.raw_attn_layout)
		self.vbox_predictions.addLayout(self.heat_attn_layout)


		self.hbox_pictures.addLayout(self.vbox_predictions)

		self.vbox1.addLayout(self.hbox_pictures)

		self.hbox2 = QHBoxLayout()
		self.hbox3 = QHBoxLayout()
		self.hbox2.addLayout(self.hbox3)

		self.ground_truth_label = QLabel()
		self.ground_truth_label.setFont(QFont("Arial", 10, QFont.Bold))
		self.ground_truth_label.setStyleSheet("color: #888888")
		self.ground_truth_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
		self.ground_truth_label.setAlignment(Qt.AlignLeft)

		self.predictions_label = QLabel()
		self.predictions_label.setFont(QFont("Arial", 10, QFont.Bold))
		self.predictions_label.setStyleSheet("color: #444444")
		self.predictions_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
		self.predictions_label.setAlignment(Qt.AlignLeft)

		self.hbox3.addWidget(self.ground_truth_label, alignment=Qt.AlignVCenter)
		self.hbox3.addWidget(self.predictions_label, alignment=Qt.AlignVCenter)

		self.predict_button = QPushButton('PREDICT', self)
		self.predict_button.setStyleSheet(
			"background-color: #8acdc2;"
			"border:0px;"
			"border-radius:45px;"
			"color: #ffffff;")
		self.predict_button.setFont(QFont("Consolas", 20, QFont.Bold))
		self.predict_button.setMinimumSize(210, 90)
		self.predict_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
		self.predict_button.clicked.connect(self.on_push_predict_button)

		self.hbox2.addWidget(self.predict_button)

		self.vbox1.addLayout(self.hbox2)

		self.hbox1.addLayout(self.vbox1)

		self.setLayout(self.hbox1)


		# predict_button.resize(predict_button.sizeHint())
		# predict_button.move(50, 50)

		self.show()

	def on_push_predict_button(self):
		# QMessageBox.information(self, "QPushButton", "Clicked predict button")
		# selected_items = self.file_list.selectedItems()
		# if len(selected_items) < 1:
		# 	return
		if self.predicting:
			return
		if self.file_name == None:
			return

		if not os.path.exists(self.save_path):
			os.makedirs(self.save_path)

		# file_name = selected_items[0].text()
		file_path = os.path.join(self.images_path, self.file_name)

		self.predict_thread = HardWorkThread(self.model_predict_function,
			file_path, self.model_param_path, self.save_path, self.file_name, resize=(224, 224), gen_hm=True)
		self.predict_thread.signal.connect(self.predict_callback)
		self.predict_thread.start()
		self.predicting = True

		self.predict_button.setText("PENDING..")
		# print(self.model_predict_function(file_path, self.model_param_path, ".", "lala", resize=(224, 224), gen_hm=True))

	def on_click_file_list(self, qListWidgetItem):
		self.file_name = qListWidgetItem.text()
		pix_map = QPixmap(self.getPath(self.file_name))
		self.preview_label.setPixmap(pix_map)
		self.preview_notation.setText(self.file_name)
		# QMessageBox.information(self, "QListView", "You have chosen: " + self.qList[qModelIndex.row()])
		self.raw_attn_img_label.setText("No prediction yet")
		self.heat_attn_img_label.setText("No prediction yet")

		self.ground_truth_label.setText(f"label of {self.file_name}")
		self.predictions_label.setText(f"pred for {self.file_name}")

	def get_file_list(self):
		file_list = os.listdir(self.images_path)
		return file_list

	def getPath(self, file_name):
		return os.path.join(self.images_path, file_name)

	def getSavePath(self, file_name):
		return os.path.join(self.save_path, file_name)

	def predict_callback(self, y_pred, true_label):
		if true_label:
			self.ground_truth_label.setText(f"true label:\n{true_label}")
		else:
			self.ground_truth_label.setText("No label received")

		if y_pred:
			pred_text = '\n'.join(f"{k}: {y_pred[k]:.5f}" for k in CATEGORIES)
			self.predictions_label.setText(pred_text)
			# self.predictions_label.setText(pred_text)
		else:
			self.predictions_label.setText("No prediction received")

		# file_name = qListWidgetItem.text()
		raw_atten_pix_map = QPixmap(self.getSavePath(f"{self.file_name}_raw_atten.jpg"))
		self.raw_attn_img_label.setPixmap(raw_atten_pix_map)
		heat_atten_pix_map = QPixmap(self.getSavePath(f"{self.file_name}_heat_atten.jpg"))
		self.heat_attn_img_label.setPixmap(heat_atten_pix_map)
		# self.predict_image_label1.setText("Image1 here")
		# self.predict_image_label2.setText("Image2 here")

		self.predict_button.setText("PREDICT")
		self.predicting = False


if __name__ == '__main__':
	# app = QApplication(sys.argv)
	# showUpWindow = GUIWindow()
	# sys.exit(app.exec_())
	from eval import predict

	app = QApplication(sys.argv)
	showUpWindow = GUIWindow(predict,
		model_param_path="../output/lala[cpu].pkl",
		# model_param_path="../output/PlantPathologyModel[cpu].pkl",
		images_path="../data/sample_img",
		save_path="./images_output")
	sys.exit(app.exec_())
