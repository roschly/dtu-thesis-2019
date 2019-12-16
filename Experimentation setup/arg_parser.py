import argparse

# TODO: is this going to fail? Does argparser have to be in main?
def arg_parser():
    ap = argparse.ArgumentParser()
    # ap.add_argument("-j", "--jobid", required=True)
    ap.add_argument("-j", "--jobid", required=False)
    args = vars(ap.parse_args())

    if args["jobid"]:
        jobid = args["jobid"]
    else:
        from datetime import datetime
        now = datetime.now()
        jobid = now.strftime("%Y-%m-%d_%H:%M:%S.%f")
    
    return jobid