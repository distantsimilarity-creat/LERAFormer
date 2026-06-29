cfg = dict(
    train = dict(
        total_epoch   = 200,
        cuda          = True,
        pretrain      = False,
        resume        = False,
        freeze_param  = False,
        ckpt_savepath = 'checkpoints/landslide4sense/',
        ckpt_resume   = r'',
        ckpt_test     = r''
    ),
    model = dict(
        model_type   = 'LERAFormer',
        num_classes  = 2
    ),
    dataset = dict(
        dataset_name = 'landslide_dataset',
        dataset_path = 'data/landslide_dataset/TrainData',
        train_lines  = 'data/landslide_dataset/TrainData/config/train.txt',
        val_lines    = 'data/landslide_dataset/TrainData/config/val.txt',
        test_lines   = 'data/landslide_dataset/TrainData/config/test.txt'
    ),
    dataloader = dict(
        isShuffle    = True,
        batch_size   = 16,
        num_workers  = 2,
        input_shape  = (128,128),
        in_channels  = 14,
        isOnLineAug  = True,
        channel_keep_idx = None,
        derived_indices = None,
        derived_replace_channels = None,
        derived_stats = None,
    ),
    optimizer=dict(
        base_lr=2e-4,
        min_lr=1e-6,
        step_size=1,
        gamma=0.9,
        weight_decay=1e-2,
        name='adamw',
        use_layer_decay=True,
        layer_decay=0.75
    ),
    scheduler=dict(
        name='cosine',
        warmup_epochs=3,
        warmup_start_factor=0.1,
        patience=10,
        factor=0.5,
        min_lr=1e-6,
    ),

    loss=dict(
        w_pos=3.0
    ),
    sampler=dict(
        enable=True,
        strategy='bucket_hard',
        bucket_bins=[0.001, 0.01, 0.05],
        bucket_weights=[0.2, 4.0, 6.0, 4.0, 2.0],
        pos_fraction=0.4,
        min_pos_per_batch=1,
        hard_factor=0.3,
        hard_momentum=0.9,
        pos_boost=2.0,
        ratio_boost=6.0,
        cache_path=None
    ),
    ema=dict(
        enable=True,
        decay=0.9999,
        eval=True,
        save=True,
        use_for_test=True
    )
)

