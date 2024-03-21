# Концепция составного обучения модели предсказанию нового состояния
## на примере задачи перевода  Text_2_Image в Text_2_Video

Основан на [предыдущих исследованиях](https://github.com/Mike030668/MIPT_magistratura/tree/main/Text2Video_project) концепции матриц вращений  

![Alt text](images/R_matrix.png)

## Базовое обучение модели для генерации определенного кадра
![Alt text](images/Normal_Back_ways_train.png)

### первые результаты
#### на малом тематическом наборе из TGIF без контроля разрешения
<img src="video/first_results/gen_normal_way_1.gif" alt="gif"  width="270"/> <img src="video/first_results/gen_normal_way_3.gif" alt="gif" width="270"/> <img src="video/first_results/gen_normal_way_2.gif" alt="gif" width="270"/> <img src="video/first_results/gen_normal_way_4.gif" alt="gif" width="270"/> <img src="video/first_results/Teddy_gitar.gif" alt="gif" width="270"/> <img src="video/first_results/Batman.gif" alt="gif" width="270"/> 
____________________________________________


# Далее показаны генерации с одинаковыми текстами и seed для сравнения
### *Генерации сделаны через 4-8 кадров, по разному. Задача пока не состояла в создании фильна, а скорее показать возможность генерации связанных кадров из стандартной и необученной на близких кадрах диффузионной модели. В данной работе используется unclip модель Kandinsky_22. Но все это применимо и к типичным диффузионным моделям*

## Модель Spliter дополняет стандартную модель Kandinsky_22, следующем образом
![Alt text](images/srtucture.png)

## Базовое обучение модели для генерации определенного кадра
#### на случайном наборе 500 видео 350-500px
<img src="video/norm_back_train/Teddy_nb.gif" alt="gif"  width="270"/> <img src="video/norm_back_train/Dog_ball_nb.gif" alt="gif" width="270"/> <img src="video/norm_back_train/Dragon_nb.gif" alt="gif" width="270"/>
____________________________________________



## Добавление матриц вращения к базовому обучению модели для генерации определенного кадра с произвольного момента
![Alt text](images/R_norm_back_train.png)

### первые результаты
#### на случайном наборе 500 видео 350-500px
<img src="video/nb_add_rote/Teddy_add_rote.gif" alt="gif"  width="270"/> <img src="video/nb_add_rote/Dog_add_rote.gif" alt="gif" width="270"/> <img src="video/nb_add_rote/Deagon_add_rote.gif" alt="gif" width="270"/>
____________________________________________



## Добавление генерации из генерации к базовому обучению модели для генерации определенного кадра с произвольного момента
![Alt text](images/Diff_norm_back_train.png)

### первые результаты
#### на случайном наборе 500 видео 350-500px
<img src="video/nb_add_diff/Teddy_add_diff (1).gif" alt="gif"  width="270"/> <img src="video/nb_add_diff/Dog_add_diff.gif" alt="gif" width="270"/> <img src="video/nb_add_diff/Drago_add_diff.gif" alt="gif" width="270"/>
____________________________________________


## Комбинированое обучение на всех техниках на каждой эпохе
### первые результаты
#### на случайном наборе 500 видео 350-500px
<img src="video/all_train/Teddy_all.gif" alt="gif"  width="270"/> <img src="video/all_train/Dog_all.gif" alt="gif" width="270"/> <img src="video/all_train/gen_from_Model_R_SpliterSimple_all_from_start_New_2.gif" alt="gif" width="270"/>
____________________________________________


## Генерация из генерации цепочкой через матрицу вращения
<img src="video/pred_from_pred/Dragon_1.gif" alt="gif"  width="270"/> <img src="video/pred_from_pred/balet_1.gif" alt="gif" width="270"/> <img src="video/pred_from_pred/Dragon_2.gif" alt="gif" width="270"/>
____________________________________________


# Просто некоторые интересные генерации
<img src="video/interesting_generation/Balet_1.gif" alt="gif"  width="400"/> <img src="video/interesting_generation/Dog_1.gif" alt="gif" width="400"/> <img src="video/interesting_generation/Dog_2.gif" alt="gif" width="400"/> <img src="video/interesting_generation/Ballet_2.gif" alt="gif" width="400"/> 

____________________________________________


### *Изучение применения матриц вращения как для тренировки сеией, так и применения внутри сети, мной только начато. Обучение построено на выучивания сеть разности в кадре как в пиксельном, так и в косинусном пространстве, поэтому требуется переход на датасет более высокогго разрешения. Далее также стоит усложнить уже саму сеть и добавить слои вращения и кросс_внимания. Кроме того, необходимо дообучать и модели Кандинского на техже датасета, для лучшей согласованной работы.*
## **Но самое важное, это то, что данный метод применим не только к текцщей задаче. Матрицы вращения, это интересный сбособ передачи косвенной информации между признаками в многомерном пространстве**
