from collections import defaultdict
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def fold_collector(dictionary: dict):
    """
    Collects  and reorders metrics from a dictionary of folds and epochs.
    """
    collected_metrics = defaultdict(list)
    for fold_num, fold in dictionary.items():
        for epoch, metrics in fold.items():
            for metric_name, metric in metrics.items():
                if metric_name == "conf_matrix":
                    continue
                if len(collected_metrics[metric_name]) <= int(fold_num):
                    collected_metrics[metric_name].append([])
                collected_metrics[metric_name][int(fold_num)].append(float(metric))
    return collected_metrics

def plot(title, metric_name:str, data, save_name, save_folder="graphs/"):
    '''Plots fold lines'''
    metric_name = metric_name.capitalize()
    plt.figure()
    for i, fold in enumerate(data):
        plt.plot(range(1, len(fold)+1), fold,  label=f"Fold {i}")
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.savefig(save_folder+save_name)
    plt.close()

def plot2(data, save_name, metric_name='Accuracy', save_folder="graphs/"):
    '''Plots averaged fold lines for each of the data sizes'''
    metric_name = metric_name.capitalize()
    plt.figure()
    order = ["0.1", "0.5", "1.0"]
    sorted_data = {key: data[key] for key in order if key in data}
    for run, series in sorted_data.items():
        plt.plot(range(1, len(series)+1), series,  label=f"Size {run}")
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.title("Average Accuracy Per Epoch For Different Data Sizes")
    plt.legend()
    plt.savefig(save_folder+save_name)
    plt.close()
    
def plot_grouped_bar_chart(allmodels_avg_data, metric_name, save_name, save_folder="graphs/"):
    '''Plots a bar chart of all averaged infomation'''
    models = list(allmodels_avg_data.keys())
    order = ["0.1", "0.5", "1.0"]
    runs = sorted(list(allmodels_avg_data[models[0]].keys()), key=lambda x: order.index(x))
    
    bar_width = 0.2
    index = range(len(runs))
    
    plt.figure()
    
    for i, model in enumerate(models):
        averages = [allmodels_avg_data[model][run][metric_name] for run in runs]
        plt.bar([x + i * bar_width for x in index], averages, bar_width, label=model)
    
    plt.xlabel('Data Size')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'Average {metric_name.capitalize()} for Each Model')
    plt.xticks([x + bar_width for x in index], runs)
    plt.legend()
    all_averages = [allmodels_avg_data[model][run][metric_name] for model in models for run in runs]
    mean_avg = np.mean(all_averages)
    std_avg = np.std(all_averages)
    plt.ylim(mean_avg - 2*std_avg, mean_avg + 2*std_avg)
    
    plt.savefig(save_folder + save_name)
    plt.close()

    
def average(data_dict):
    '''Averages the last two metric values in a list-nested dictionary'''
    avg_data = {}
    for metric_name, data in data_dict.items():
        d = [sum(fold[-2:]) / 2 for fold in data]
        avg_data[metric_name] = sum(d)/len(d)
    return avg_data


folder_names = ["Bert/", "DistilBert/", "XLNet/"]
files = {}
for folder in folder_names:
    files[folder[0:-1].lower()] = os.listdir(folder)

allmodels_avg_data = defaultdict(lambda: defaultdict(dict))
folds_avg_data = {}

models = ["BERT", "DistilBERT", "XLNet"]
for model, folder in zip(models, folder_names):
    metrics_dict = {}
    for run in files[model.lower()]:
        path = f"{folder}{run}/metrics/metrics.json"
        with open(path, 'r') as file:
            metrics_data = json.load(file)
            metrics_dict[run[-3:]] = metrics_data

    data_dict = {}
    compare_runs = {}
    for run, data in metrics_dict.items():
        data_dict[run] = fold_collector(data)
        for metric, data in data_dict[run].items():
            plot(title=f"{model} {run}-Size {metric.capitalize()}",metric_name=metric, data=data, save_name=f"{model}_{run}_{metric}.png")
        allmodels_avg_data[model][run] = average(data_dict[run])
        zipped = list(zip(*data_dict[run]['accuracy']))
        compare_runs[run] = [sum(inner_list) / len(inner_list) for inner_list in zipped]
    
    plot2(data=compare_runs, save_name=f"{model}_grouped_accuracy.png")
        


for metric in ["accuracy", "f1", "precision", "recall"]:
    plot_grouped_bar_chart(allmodels_avg_data, metric_name=metric, save_name=f"grouped_bar_chart_{metric}.png")







        

    
    
        
        
                
            
            


        
        
        
        
        
        


        
        
        

        
        