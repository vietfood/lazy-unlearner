from rmu.config import UnlearnConfig
from rmu.model_utils import get_data
from rmu.unlearn import UnlearnTrainer
from rmu.utils import cleanup_wandb, init_wandb
from simple_parsing import parse

if __name__ == "__main__":
    args = parse(UnlearnConfig)
    init_wandb(args, project_name="RMU_Unlearn_Test")

    # ----- Load data -----
    forget_data_list, retain_data_list = get_data(
        args.forget_corpora,
        args.retain_corpora,
        args.min_len,
        args.batch_size,
    )

    # ----- Training -----
    trainer = UnlearnTrainer(
        forget_data=forget_data_list,
        retain_data=retain_data_list,
        args=args,
        trial=None,
    )
    trainer.run(save=False)

    # ---- Clean up -----
    cleanup_wandb()
