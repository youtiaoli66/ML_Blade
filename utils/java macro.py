import os
import csv
from tqdm import tqdm

def generate_macro(path):
    filenames=os.listdir(path)
    for name in tqdm(filenames,'read data...'):
        with open(path+'/'+name+'/STARCCM_3/output_data.java','w',encoding='utf-8',newline='') as file1:
           file1.write(
               '''
               // Simcenter STAR-CCM+ macro: Output_data.java
// Written by Simcenter STAR-CCM+ 19.02.009
package macro;

import java.util.*;

import star.common.*;
import star.base.neo.*;

public class Output_data extends StarMacro {

  public void execute() {
    execute0();
  }

  private void execute0() {

    Simulation simulation_0 = 
      getActiveSimulation();

    XyzInternalTable xyzInternalTable_0 = 
      simulation_0.getTableManager().create("star.common.XyzInternalTable");

    PrimitiveFieldFunction primitiveFieldFunction_0 = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("MachNumber"));

    xyzInternalTable_0.setFieldFunctions(new ArrayList<>(Arrays.<FieldFunction>asList(primitiveFieldFunction_0)));

    PrimitiveFieldFunction primitiveFieldFunction_1 = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("Pressure"));

    PrimitiveFieldFunction primitiveFieldFunction_2 = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("Velocity"));

    VectorComponentFieldFunction vectorComponentFieldFunction_0 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_2.getComponentFunction(0));

    VectorComponentFieldFunction vectorComponentFieldFunction_1 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_2.getComponentFunction(1));

    VectorComponentFieldFunction vectorComponentFieldFunction_2 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_2.getComponentFunction(2));

    xyzInternalTable_0.setFieldFunctions(new ArrayList<>(Arrays.<FieldFunction>asList(primitiveFieldFunction_0, primitiveFieldFunction_1, vectorComponentFieldFunction_0, vectorComponentFieldFunction_1, vectorComponentFieldFunction_2)));

    xyzInternalTable_0.getParts().setQuery(null);

    Region region_0 = 
      simulation_0.getRegionManager().getRegion("BLADE");

    Boundary boundary_0 = 
      region_0.getBoundaryManager().getBoundary("FACES");

    Boundary boundary_1 = 
      region_0.getBoundaryManager().getBoundary("INLET");

    InterfaceBoundary interfaceBoundary_0 = 
      ((InterfaceBoundary) region_0.getBoundaryManager().getBoundary("INLET [BLADE/HEAD]"));

    Boundary boundary_2 = 
      region_0.getBoundaryManager().getBoundary("OUTLET");

    InterfaceBoundary interfaceBoundary_1 = 
      ((InterfaceBoundary) region_0.getBoundaryManager().getBoundary("OUTLET [BLADE/TAIL]"));

    Boundary boundary_3 = 
      region_0.getBoundaryManager().getBoundary("PER1");

    InterfaceBoundary interfaceBoundary_2 = 
      ((InterfaceBoundary) region_0.getBoundaryManager().getBoundary("PER1 [BLADE/BLADE]"));

    Boundary boundary_4 = 
      region_0.getBoundaryManager().getBoundary("PER2");

    InterfaceBoundary interfaceBoundary_3 = 
      ((InterfaceBoundary) region_0.getBoundaryManager().getBoundary("PER2 [BLADE/BLADE]"));

    xyzInternalTable_0.getParts().setObjects(region_0, boundary_0, boundary_1, interfaceBoundary_0, boundary_2, interfaceBoundary_1, boundary_3, interfaceBoundary_2, boundary_4, interfaceBoundary_3);

    xyzInternalTable_0.extract();''')
           file1.write('''
    xyzInternalTable_0.export("C://Users//86176//Desktop//python//AICFD//heeds//heeds_Study_1//HEEDS_0//''')
           file1.write(f'Design{name[6:]}')
           file1.write('''//STARCCM_3//XYZ Internal Table.csv", ",");''')
           file1.write('''

  }
}
               '''

           )

    for name in tqdm(filenames,'read data...'):
        with open(path+'/'+name+'/STARCCM_3/output_blade_boundary.java','w',encoding='utf-8',newline='') as file2:
           file2.write(
               '''
// Simcenter STAR-CCM+ macro: output_blade_boundary.java
// Written by Simcenter STAR-CCM+ 19.02.009
package macro;

import java.util.*;

import star.common.*;
import star.base.neo.*;

public class output_blade_boundary extends StarMacro {

  public void execute() {
    execute0();
  }

  private void execute0() {

    Simulation simulation_0 = 
      getActiveSimulation();

    XyzInternalTable xyzInternalTable_0 = 
      simulation_0.getTableManager().create("star.common.XyzInternalTable");

    PrimitiveFieldFunction primitiveFieldFunction_0 = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("MachNumber"));

    PrimitiveFieldFunction primitiveFieldFunction_1 = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("Pressure"));

    PrimitiveFieldFunction primitiveFieldFunction_2 = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("Velocity"));

    VectorComponentFieldFunction vectorComponentFieldFunction_0 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_2.getComponentFunction(0));

    VectorComponentFieldFunction vectorComponentFieldFunction_1 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_2.getComponentFunction(1));

    VectorComponentFieldFunction vectorComponentFieldFunction_2 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_2.getComponentFunction(2));

    xyzInternalTable_0.setFieldFunctions(new ArrayList<>(Arrays.<FieldFunction>asList(primitiveFieldFunction_0, primitiveFieldFunction_1, vectorComponentFieldFunction_0, vectorComponentFieldFunction_1, vectorComponentFieldFunction_2)));

    xyzInternalTable_0.getParts().setQuery(null);

    Region region_0 = 
      simulation_0.getRegionManager().getRegion("BLADE");

    Boundary boundary_0 = 
      region_0.getBoundaryManager().getBoundary("FACES");

    xyzInternalTable_0.getParts().setObjects(boundary_0);

    xyzInternalTable_0.setPresentationName("XYZ blade boundary");

    xyzInternalTable_0.extract();

    ''')
           file2.write('''
    xyzInternalTable_0.export("C://Users//86176//Desktop//python//AICFD//heeds//heeds_Study_1//HEEDS_0//''')
           file2.write(f'Design{name[6:]}')
           file2.write('''//STARCCM_3//XYZ blade boundary.csv", ",");''')
           file2.write('''

  }
}
               '''

           )

    for name in tqdm(filenames,'read data...'):
       with open(path+'/'+name+'/STARCCM_3/output_side_boundary.java','w',encoding='utf-8',newline='') as file3:
           file3.write(
               '''
             // Simcenter STAR-CCM+ macro: output_side_boundary.java
// Written by Simcenter STAR-CCM+ 19.02.009
package macro;

import java.util.*;

import star.common.*;
import star.base.neo.*;

public class output_side_boundary extends StarMacro {

  public void execute() {
    execute0();
  }

  private void execute0() {

    Simulation simulation_0 = 
      getActiveSimulation();

    XyzInternalTable xyzInternalTable_1 = 
      simulation_0.getTableManager().create("star.common.XyzInternalTable");

    xyzInternalTable_1.setPresentationName("XYZ side boundary");

    PrimitiveFieldFunction primitiveFieldFunction_0 = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("MachNumber"));

    xyzInternalTable_1.setFieldFunctions(new ArrayList<>(Arrays.<FieldFunction>asList(primitiveFieldFunction_0)));

    PrimitiveFieldFunction primitiveFieldFunction_1 = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("Pressure"));

    PrimitiveFieldFunction primitiveFieldFunction_2 = 
      ((PrimitiveFieldFunction) simulation_0.getFieldFunctionManager().getFunction("Velocity"));

    VectorComponentFieldFunction vectorComponentFieldFunction_0 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_2.getComponentFunction(0));

    VectorComponentFieldFunction vectorComponentFieldFunction_1 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_2.getComponentFunction(1));

    VectorComponentFieldFunction vectorComponentFieldFunction_2 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_2.getComponentFunction(2));

    xyzInternalTable_1.setFieldFunctions(new ArrayList<>(Arrays.<FieldFunction>asList(primitiveFieldFunction_0, primitiveFieldFunction_1, vectorComponentFieldFunction_0, vectorComponentFieldFunction_1, vectorComponentFieldFunction_2)));

    xyzInternalTable_1.getParts().setQuery(null);

    Region region_0 = 
      simulation_0.getRegionManager().getRegion("BLADE");

    Boundary boundary_1 = 
      region_0.getBoundaryManager().getBoundary("INLET");

    InterfaceBoundary interfaceBoundary_0 = 
      ((InterfaceBoundary) region_0.getBoundaryManager().getBoundary("INLET [BLADE/HEAD]"));

    Boundary boundary_2 = 
      region_0.getBoundaryManager().getBoundary("OUTLET");

    InterfaceBoundary interfaceBoundary_1 = 
      ((InterfaceBoundary) region_0.getBoundaryManager().getBoundary("OUTLET [BLADE/TAIL]"));

    Boundary boundary_3 = 
      region_0.getBoundaryManager().getBoundary("PER1");

    InterfaceBoundary interfaceBoundary_2 = 
      ((InterfaceBoundary) region_0.getBoundaryManager().getBoundary("PER1 [BLADE/BLADE]"));

    Boundary boundary_4 = 
      region_0.getBoundaryManager().getBoundary("PER2");

    InterfaceBoundary interfaceBoundary_3 = 
      ((InterfaceBoundary) region_0.getBoundaryManager().getBoundary("PER2 [BLADE/BLADE]"));

    xyzInternalTable_1.getParts().setObjects(boundary_1, interfaceBoundary_0, boundary_2, interfaceBoundary_1, boundary_3, interfaceBoundary_2, boundary_4, interfaceBoundary_3);

    xyzInternalTable_1.extract();

    
''')
           file3.write('''
    xyzInternalTable_1.export("C://Users//86176//Desktop//python//AICFD//heeds//heeds_Study_1//HEEDS_0//''')
           file3.write(f'Design{name[6:]}')
           file3.write('''//STARCCM_3//XYZ side boundary.csv", ",");''')
           file3.write('''

  }
}
               '''

           )

if __name__ == '__main__':
    generate_macro('C:/Users/86176/Desktop/python/AICFD/heeds/heeds_Study_1/HEEDS_0')