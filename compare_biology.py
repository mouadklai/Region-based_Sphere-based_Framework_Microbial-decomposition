from utils import compare_bio as Compare


if __name__ == "__main__":
    #Parameters 
    root = 'Plots and Results/'
    print("comparison of results from biology simulations  :: ")
    file1 = root+input( "File 1 -->  Enter a file from 'Plots and Results' folder   :: ")
    discription1 = input("Enter a description of the file ("+file1+") 'the description will be seen in the figure'   : ")
    file2 = root+input( "File 2 -->  Enter a file from 'Plots and Results' folder   :: ")
    discription2 = input("Enter a description of the file ("+file2+") 'the description will be seen in the figure'   : ")
    simulation_time = float(input("simulation time in (days)   ::"))
    comparison_experience  = input("Could you provide a description for this comparison 'the description will be seen in the figure'")
    Compare(file1,file2,discription1,discription2,simulation_time,comparison_experience)
