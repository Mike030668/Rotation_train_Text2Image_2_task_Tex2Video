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


## Генерация из генерации цепрчкой
<img src="video/pred_from_pred/Dragon_1.gif" alt="gif"  width="270"/> <img src="video/pred_from_pred/balet_1.gif" alt="gif" width="270"/> <img src="video/pred_from_pred/Dragon_2.gif" alt="gif" width="270"/>
