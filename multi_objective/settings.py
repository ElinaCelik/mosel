import random
import os
# datsets override methods override generic
#
# datasets
#
adult = dict(
    dataset='adult',
    dim=(88,),
    objectives=['BinaryCrossEntropyLoss', 'deo'],
    epochs=50,
    use_scheduler=False,
    lamda=.01,
    alpha=.5,
)

credit = dict(
    dataset='credit',
    dim=(90,),
    objectives=['BinaryCrossEntropyLoss', 'deo'],
    epochs=50,
    use_scheduler=False,
    lamda=.01,
    alpha=[.1, .5],
)

compass = dict(
    dataset='compass',
    dim=(20,),
    objectives=['BinaryCrossEntropyLoss', 'deo'],
    epochs=50,
    use_scheduler=False,
    lamda=.01,
    alpha=.5,
)

multi_mnist = dict(
    dataset='multi_mnist',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    lamda=8,
    alpha=1.2,
)

multi_fashion = dict(
    dataset='multi_fashion',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    lamda=2,
    alpha=1.2,
)

multi_fashion_mnist = dict(
    dataset='multi_fashion_mnist',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    lamda=8,
    alpha=1.2,
)

celeba = dict(
    dataset='celeba',
    dim=(3, 64, 64),
    task_ids=[16, 22],                                    # easy tasks
    # task_ids=[25, 27],                                    # hard tasks
    # task_ids=[16, 22, 24],                                # 3 objectives
    # task_ids=[26, 22, 24, 26],                            # 4 objectives
    # task_ids=[random.randint(0, 39) for _ in range(10)],  # 10 random tasks
    objectives=['BinaryCrossEntropyLoss' for _ in range(2)],
    #n_test_rays=1,
    epochs=30,
    use_scheduler=False,
    train_eval_every=10,   
    eval_every=10,           
    model_name='efficientnet-b4',
    lr=0.0005,
    lamda=3,
    alpha=1,
    checkpoint_every=1,
    batch_size=32,
)

fairface = dict(
    dataset='fairface',
    dim=(3, 224, 224),
    task_ids=[1,1],
    objectives=['BinaryCrossEntropyLoss', 'deo'],
    epochs=30,
    use_scheduler=True,
    train_eval_every=10,     
    eval_every=10,           
    model_name='resnet34',
    lr=0.0001,
    lamda=0.1,
    alpha=0.5,
    checkpoint_every=1,
    batch_size=128,
)

civil_comments = dict(
    dataset='civil_comments',
    dim=(0,),
    text=True,
    task_ids=[0, 0],
    objectives=['BinaryCrossEntropyLoss', 'deo'],
    epochs=3,                  
    use_scheduler=False,
    model_name='distilbert-base-uncased',
    lr=[2e-6, 2e-5],                        
    batch_size=64,           
    max_length=128,               
    max_samples=100000,
    alpha=0.5,               
    lamda=0.01,                    
    reference_point=[2,2], 
    checkpoint_every=1,
    eval_every=1,
    train_eval_every=1,
)

#
# methods
#
paretoMTL = dict(
    method='ParetoMTL',
    num_starts=5,
    scheduler_gamma=0.5,
    scheduler_milestones=[15,30,45,60,75,90],
)

cosmos = dict(
    method='cosmos',
    lamda=2,        # Default for multi-mnist
    alpha=1.2,      
)

mosel = dict(
    method='mosel',
    alpha=1.2,
    lr = [1e-4,1e-3],
    split_network=2,
    scalarization = 'ls',
)

mgda = dict(
    method='mgda',
    lr=1e-4,
    approximate_norm_solution=False,
    normalization_type='loss+',
    use_scheduler=False
)

SingleTaskSolver = dict(
    method='SingleTask',
    num_starts=2,   # two times for two objectives (sequentially)
)

uniform_scaling = dict(
    method='uniform',
)

hyperSolver_ln = dict(
    method='hyper_ln',
    lr=1e-4,
    epochs=150,
    alpha=.2,   # dirichlet sampling
    use_scheduler=False,
    internal_solver='linear', # 'epo' or 'linear'
)

hyperSolver_epo = dict(
    method='hyper_epo',
    lr=1e-4,
    epochs=150,
    alpha=.2,   # dirichlet sampling
    use_scheduler=False,
    internal_solver='epo',
)

fair_classifier = dict(
    method='fair_classifier',
    lmda=0.5,
    lr=1e-4,
    n_test_rays=25,
    baseline = False
)

pcgrad = dict(
    method='pcgrad',
    lr=1e-3,
    num_starts=5
)

gradnorm = dict(
    method='gradnorm',
    lr=1e-3,
    asymmetry=1.5,
)

nashmtl = dict(
    method='nashmtl',
    lr=1e-3,
    max_norm=1.0,
    update_weights_every=1,
)

famo = dict(
    method='famo',
    lr=1e-3,
    w_lr=0.025,
    w_weight_decay=1e-5,
)

#
# Common settings
#
generic = dict(    
    # Seed.
    seed=1,
    
    # Directory for logging the results
    #logdir=os.path.join(os.environ['HOME'], 'results'),
    logdir='results',

    # dataloader worker threads
    num_workers=4,

    # Number of test preference vectors for Pareto front generating methods    
    n_test_rays=25,

    # Evaluation period for val and test sets (0 for no evaluation)
    eval_every=10,

    # Evaluation period for train set (0 for no evaluation)
    train_eval_every=10,

    # Checkpoint period (0 for no checkpoints)
    checkpoint_every=10,

    # Use a multi-step learning rate scheduler with defined gamma and milestones
    use_scheduler=True,
    scheduler_gamma=0.1,
    scheduler_milestones=[20,40,80,90],

    # Number of train rays for methods that follow a training preference (ParetoMTL and MGDA)
    num_starts=1,

    # Training parameters
    lr=1e-3,
    batch_size=256,
    epochs=100,

    # Reference point for hyper-volume calculation
    reference_point=[1, 1],
    text=False
)
