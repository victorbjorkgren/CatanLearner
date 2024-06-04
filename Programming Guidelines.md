**Programming Guidelines**

*Module Order*

1. Module Docstring
Start with a module-level docstring that explains the purpose and content of the module. This is good practice for readability and maintainability.

2. Imports

	List all your import statements next, grouped in the following order:
   - Standard library imports
     - Related third party imports
     - Local application/library specific imports
     Make sure to separate these groups by a blank line.

3. Constants
Define any constants that your class or classes within the module might use.
Capital letters for Magic Numbers

4. Classes
For each class, you can follow this order:


*Class Order*

1.  *Class Docstring*

	Start with a docstring immediately after the class definition to describe the purpose and behavior of the class.

2. *Class Variables*

	- Public Class Variables: Start with variables that are shared across all instances of a class.

	- Private Class Variables (_var): Follow with variables that should not be accessed directly outside the class.
	
3. *_ _ init _ _*

	- Public Instance Variables: Define these in your _ _ init _ _ method or other methods as needed.

	- Private Instance Variables (_var and _ _var): Private variables intended for internal class use only. Prefer these
	over public methods

4. *Magic Methods*

	After _ _ init _ _ list other dunder methods like _ _ str _ _, _ _ repr _ _, _ _ len _ _, etc. These methods should be grouped for readability.

5. *Properties* 

	@property and Setters

6. *Metods*
	- Public Methods: Define these next. They are the methods intended to be accessed from outside the class.
	- Private Methods (_method and _ _method): Methods for internal use within the class. Start with _method for protected methods and __method for private methods.
	- Static Methods (@staticmethod): Methods that don't access class or instance-specific data.
	- Class Methods (@classmethod): Methods that access class variables but not instance variables.


**Functions**

Do Type checking of all variables and returns. Thank me later.

	

**Final Notes**
	
Use PEP-8 as far as feasible	

Ordering within sections: Within each section (methods, variables), you might further organize by visibility (public before private) and then alphabetically or logically (grouping methods by functionality or use case).

Consistency is key: Whatever structure you choose, the most important aspect is consistency across your project. This makes it easier for you and others to understand and maintain the code.

