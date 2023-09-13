#**********************************************************
# GPU Information Display
#**********************************************************

import torch

def displayGpuStatus(showFlag):
    totalGpuNum = torch.cuda.device_count()

    if showFlag :
        print("="*80)
        if totalGpuNum  > 0 :
            print("Available GPU Devices: ")
            for i in range(totalGpuNum):
                print("[[[ Device [%d]: %s ]]]" %(i, torch.cuda.get_device_properties(i)))
            print("-"*50)
                
            print("Current GPU being used: ",torch.cuda.current_device())
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        else :
            print("NO Gpu exists...... Will use the CPU")
        print("="*80)
        
    return totalGpuNum
