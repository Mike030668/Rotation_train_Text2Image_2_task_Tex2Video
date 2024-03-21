# Концепция составного обучения модели предсказанию нового состояния
## на примере задачи перевода  Text_2_Image в Text_2_Video

Основан на [предыдущих исследованиях](https://github.com/Mike030668/MIPT_magistratura/tree/main/Text2Video_project) концепции матриц вращений  

![Alt text](images/R_matrix.png)

## Базовое обучение модели для генерации определенного кадра
![Alt text](images/Normal_Back_ways_train.png)

### первые результаты
#### на малом тематическом наборе
<img src="video/first_results/gen_normal_way_1.gif" alt="gif"  width="230"/> <img src="video/first_results/gen_normal_way_3.gif" alt="gif" width="230"/> <img src="video/first_results/gen_normal_way_2.gif" alt="gif" width="230"/> <img src="video/first_results/gen_normal_way_4.gif" alt="gif" width="230"/> 

#### на случайном наборе 500 видео 350-500px
<img src="video/norm_back_train/Teddy_nb.gif" alt="gif"  width="250"/> <img src="video/norm_back_train/Dog_ball_nb.gif" alt="gif" width="250"/> <img src="video/norm_back_train/Dragon_nb.gif" alt="gif" width="250"/>

## Добавление матриц вращения к базовому обучению модели для генерации определенного кадра с произвольного момента
![Alt text](images/R_norm_back_train.png)

### первые результаты
