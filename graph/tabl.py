import matplotlib.pyplot as plt
import numpy as np

def create_metrics_chart():
    
    # Ваши данные
    models = ['SSD', 'Faster R-CNN', 'YOLO']
    precision = [0.8557, 0.7368, 0.8586]
    recall = [0.4955, 0.8358, 0.4896]
    f1 = [0.6276, 0.7832, 0.6236]
    
    # Настройка позиций столбцов
    x = np.arange(len(models))
    width = 0.25
    
    # Создание фигуры
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Столбцы
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498DB', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#E74C3C', edgecolor='black', linewidth=1)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#2ECC71', edgecolor='black', linewidth=1)
    
    # Настройка графика
    ax.set_xlabel('Модель', fontsize=12, weight='bold')
    ax.set_ylabel('Значение метрики', fontsize=12, weight='bold')
    ax.set_title('Сравнение метрик моделей детекции', fontsize=14, weight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, weight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Добавление значений на столбцы
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, weight='bold')
    
    # Легенда
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black')
    
    # Добавление рамки
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    
    plt.tight_layout()
    plt.savefig('model_metrics_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# Запуск
create_metrics_chart()