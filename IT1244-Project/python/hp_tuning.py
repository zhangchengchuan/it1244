import os
import subprocess

for batch_size in [128]:
    for learning_rate in [0.01, 0.001]:
            for dropout in [0.2, 0.4, 0.6]:
                for model in ['cnn']:
                    # Run the entire pipeline
                    label = f'{model}_lr-{learning_rate}_bs-{batch_size}_do-{dropout}'
                    train_loc = './Dataset/Audio Dataset/Cats and Dogs/data'
                    test_loc = './Dataset/Audio Dataset/Cats and Dogs/test_data'
                    folder_loc = './models/'
                    main_command = f"python python/main.py -t '{model}' -lr {learning_rate} -bs {batch_size} -do {dropout}"
                    evaluation_command = f"python python/evaluation.py -t '{model}' -lr {learning_rate} -bs {batch_size} -do {dropout}"

                    try:
                         print(f"Running: {main_command}")
                         subprocess.run(main_command, check=True, shell=True)

                         print(f"Running: {evaluation_command}")
                         subprocess.run(evaluation_command, check=True, shell=True)
                    except Exception as e:
                         print(f"Failed to run commands: {e}")
                         continue
                    finally:
                        print("\n")

                    


