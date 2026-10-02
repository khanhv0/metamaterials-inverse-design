import subprocess
import os
import csv

from odbAccess import openOdb


def extract_properties(path):
    RVE_volume = 400
    strain = 10 / (RVE_volume ** 0.5)

    # Open odb
    odb = openOdb(path=path + "/Job.odb")

    # Use last frame of first step
    stepName = odb.steps.keys()[0]
    frame = odb.steps[stepName].frames[-1]

    # Stress and integration volume fields
    stress = frame.fieldOutputs['S']
    ivol = frame.fieldOutputs['IVOL']

    # Build dictionary for IVOL: (element, integration point) -> volume
    ivolDict = {(v.elementLabel, v.integrationPoint): v.data for v in ivol.values}

    # Initialize accumulators
    weighted = {"S11": 0.0, "S22": 0.0, "S12": 0.0}
    total_volume = 0.0

    # Loop over stress values and weight with IVOL
    for v in stress.values:
        key = (v.elementLabel, v.integrationPoint)
        if key in ivolDict:
            vol = ivolDict[key]
            s11, s22, s12 = v.data[0], v.data[1], v.data[2]
            weighted["S11"] += s11 * vol
            weighted["S22"] += s22 * vol
            weighted["S12"] += s12 * vol
            total_volume += vol

    odb.close()
    # Compute averages
    if total_volume > 0:
        avg = {k: weighted[k] / RVE_volume for k in weighted}

        poisson = avg["S22"] / avg["S11"]

        young = (avg["S11"] - poisson * avg["S22"]) / strain

        return [poisson, young, total_volume / RVE_volume]
    else:
        print("No volumes found.")
        return [0, 0, 0]


dir_path = "./batch_results/"
subdirs = [os.path.join(dir_path, name) for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
success = 0

with open("test_samples.csv", "rb") as param_f:
    with open("simulation_results.csv", "wb") as result_f:
        reader = csv.reader(param_f)
        writer = csv.writer(result_f)

        writer.writerow(next(reader) + ["poisson", "young", "volume_frac"])

        for index, subdir in enumerate(subdirs):
            name = subdir.split("/")[-1]
            print("Evaluating {}".format(name))

            if os.path.exists(subdir + "/Job.lck"):
                os.remove(subdir + "/Job.lck")

            if not os.path.exists(subdir + "/Job.odb"):
                abs_path = os.path.abspath(subdir)
                simulation_cmd = "abaqus job=Job input={0}.inp interactive".format(name)
                subprocess.call(simulation_cmd, cwd=abs_path, shell=True)

            try:
                writer.writerow(next(reader) + extract_properties(subdir))
                success += 1
            except:
                pass

print("Success {}/1000".format(success))
