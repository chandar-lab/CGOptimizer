# generate run scripts
import argparse
import glob
import os
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--comet_api", type=str, default='')
    parser.add_argument("--comet_workspace", type=str, default='koustuvs')
    parser.add_argument("--comet_project", type=str, default='compositionality-nli')
    parser.add_argument('--loc', type=str, default='dataset/compositionality/quantifiers/koustuv/cmb/')
    parser.add_argument('--models', type=str, default='InferSent,BLSTMprojEncoder,BGRUlastEncoder,InnerAttentionMILAEncoder,InnerAttentionYANGEncoder,InnerAttentionNAACLEncoder,ConvNetEncoder,LSTMEncoder')
    parser.add_argument('--local', action='store_true', help="If true, run on machines not on slurm")
    parser.add_argument('--gpus',type=str, default='0', help='works in local, run jobs on these gpus')
    parser.add_argument('--only', type=str, default='', help='run only these experiment headers')
    parser.add_argument('--stdout', type=str, default='std_outputs', help='folder to store std outputs')
    parser.add_argument('--seed', type=int, default=1111, help='seed value')

    args = parser.parse_args()

    models = args.models.split(',')
    gpus = args.gpus.split(',')
    if not os.path.exists(args.stdout):
        os.makedirs(args.stdout)
    if not args.local and len(gpus) > 1:
        print("set gpus to 0 for non local jobs")
        exit(0)
    folders = glob.glob(args.loc + '*/')
    print("Found {} folders".format(len(folders)))
    ct = 0
    run_flnames = {gpu:[] for gpu in gpus}
    for folder in folders:
        print("Directory : {}".format(folder))
        if len(args.only) > 0:
            if args.only not in folder:
                print("Skipping this.")
                continue
        gpu_choice = random.choice(gpus)
        run_file = "#!/bin/sh\n"
        last_path = folder.split('/')[-2]
        exp_folder = 'exp_seed_{}'.format(args.seed)
        if not os.path.exists(exp_folder):
            os.mkdir(exp_folder)
        save_folder_name = os.path.join(exp_folder, last_path)
        if not os.path.exists(save_folder_name):
            os.mkdir(save_folder_name)
        if not args.local:
            run_file += "#SBATCH --time=0-6:00\n"
            run_file += "#SBATCH --account=rrg-dprecup\n"
            run_file += "#SBATCH --nodes=1\n"
            run_file += "#SBATCH --ntasks=16\n"
            run_file += "#SBATCH --gres=gpu:1\n"
            run_file += "#SBATCH --mem=0\n"
            run_file += "#SBATCH --output={}/slurm_%j.out\n".format(save_folder_name)
            run_file += "#SBATCH --error={}/slurm_%j.err\n".format(save_folder_name)
            run_file += "module load nixpkgs/16.09\n"
            run_file += "module load intel/2018.3\n"
            run_file += "module load cuda/10.0.130\n"
            run_file += "module load cudnn/7.4\n"
            run_file += "source activate compnli\n"
            run_file += "cd /home/koustuvs/projects/def-jpineau/koustuvs/InferSent-comp/\n"
        run_file += "export COMET_API='{}'\n".format(args.comet_api)
        run_file += "export COMET_WORKSPACE='{}'\n".format(args.comet_workspace)
        run_file += "export COMET_PROJECT='{}'\n".format(args.comet_project)
        run_file += "\n"

        for model in models:
            if args.local:
                pre = 'CUDA_VISIBLE_DEVICES={} '.format(gpu_choice)
            else:
                pre = ''
            base_path = folder.split('/')[-2]
            std_output_folder = os.path.join(exp_folder, args.stdout)
            if not os.path.exists(std_output_folder):
                os.mkdir(std_output_folder)
            output_file = os.path.join(std_output_folder,'{}_{}.out'.format(base_path, model))
            run_file += pre + "python train_nli.py --nlipath {} --encoder_type {} --optimizer adam,lr=0.001 --lrshrink 0.9 --enc_lstm_dim 300 --word_emb_type normal --seed {} --comet_apikey $COMET_API --comet_workspace $COMET_WORKSPACE --comet_project $COMET_PROJECT > {}\n".format(folder, model, args.seed, output_file)
            ct += 1
        run_file += "\n"

        print("Writing file")
        run_folder = os.path.join(exp_folder, "runs")
        if not os.path.exists(run_folder):
            os.makedirs(run_folder)
        last_path = folder.split('/')[-2]
        run_flname = os.path.join(run_folder, 'run_{}.sh'.format(last_path))
        with open(run_flname, 'w') as fp:
            fp.write(run_file)
        run_flnames[gpu_choice].append(run_flname)
    print("Done, now writing the meta runner")
    for gpu in gpus:
        meta_file = "#!/bin/sh\n"
        for rf in run_flnames[gpu]:
            if not args.local:
                meta_file += "sbatch {}\n".format(rf)
            else:
                meta_file += "./{}\n".format(rf)
        with open('meta_run_{}.sh'.format(gpu),'a') as fp:
            fp.write(meta_file)
    print("Number of experiments to run : {}".format(ct))





