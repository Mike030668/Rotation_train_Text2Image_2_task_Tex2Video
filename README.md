# Концепция составного обучения модели предсказанию нового состояния
## на примере задачи перевода  Text_2_Image в Text_2_Video

Основан на [предыдущих исследованиях](https://github.com/Mike030668/MIPT_magistratura/tree/main/Text2Video_project) концепции матриц вращений  

![Alt text](images/R_matrix.png)

## Базовое обучение модели для генерации определенного кадра
![Alt text](images/Normal_Back_ways_train.png)

### первые результаты
#### на малом тематическом наборе из TGIF без контроля разрешения
<img src="video/first_results/gen_normal_way_1.gif" alt="gif"  width="250"/> <img src="video/first_results/gen_normal_way_3.gif" alt="gif" width="250"/> <img src="video/first_results/gen_normal_way_2.gif" alt="gif" width="250"/> <img src="video/first_results/gen_normal_way_4.gif" alt="gif" width="250"/> <img src="video/first_results/Teddy_gitar.gif" alt="gif" width="250"/> <img src="vvideo/first_results/Batman.gif" alt="gif" width="250"/> 

#### на случайном наборе 500 видео 350-500px
<img src="video/norm_back_train/Teddy_nb.gif" alt="gif"  width="270"/> <img src="video/norm_back_train/Dog_ball_nb.gif" alt="gif" width="270"/> <img src="video/norm_back_train/Dragon_nb.gif" alt="gif" width="270"/>

## Добавление матриц вращения к базовому обучению модели для генерации определенного кадра с произвольного момента
![Alt text](images/R_norm_back_train.png)

### первые результаты
#### на случайном наборе 500 видео 350-500px
<img src="video/nb_add_rote/Teddy_add_rote.gif" alt="gif"  width="270"/> <img src="video/nb_add_rote/Dog_add_rote.gif" alt="gif" width="270"/> <img src="video/nb_add_rote/Deagon_add_rote.gif" alt="gif" width="270"/>
