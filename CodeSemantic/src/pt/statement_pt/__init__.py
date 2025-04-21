from ..abst_pt import AbstPt
import sys
sys.path.append('/home/monoshi/CodeSemantic/CodeSemantic')
from dataset_utils import incontext_shots_with_same_statement


class StatementPt1(AbstPt):
    def __init__(self, name, demos=None, args=None):
        super().__init__(name, demos or [])
        self.args = args
        self.demos = demos
        self.pt_template_Assignment = \
            ("Given the following {lang} code snippet and the selected statement, "
             "the local variable values before the statements are shown as follows, "
             "what will be the value of the selected statement after executing the selected statement?\n\n"
             "Code Snippet\n"
             "```{lang}\n"
             "{code}\n"
             "```\n\n"
             "Selected Statement: {statement}\n\n"
             "Local Variables:\n"
             "{variables}\n\n"
             "Please put your answer in the <ans></ans> tags"
             )
        
        self.pt_template_Branch = \
            ("Given the following {lang} code snippet and the selected branch statement, "
            "the local variable values before the branch statements are shown as follows, "
            "Will the nvidbranch be executed based on the condition expression variable values? Please answer \"Yes\" or \"No\".\n\n"
            "Code Snippet\n"
            "```{lang}\n"
            "{code}\n"
            "```\n\n"
            "Selected Branch Statement: {statement}\n\n"
            "If Expression Variables:\n"
            "{variables}\n\n"
            "Please put your answer in the <ans></ans> tags"
            )

        self.pt_template_API = \
            ("Given the following {lang} code snippet and the selected statement, "
             "the local variable values of the api/function parameters are shown as follows, "
             "what will be the value after the selected API/Function call?\n\n"
             "Code Snippet\n"
             "```{lang}\n"
             "{code}\n"
             "```\n\n"
             "Selected API/Function: {statement}\n\n"
             "API/Function Parameters:\n"
             "{variables}\n\n"
             "Please put your answer in the <ans></ans> tags"
             )
        
        self.pt_template_block = \
            ("Given the following {lang} code snippet and the selected statement, "
             "the input of the code snippet is given as follows, "
             "what will be the value after executing the selected statement?\n\n"
             "Code Snippet\n"
             "```{lang}\n"
             "{code}\n"
             "```\n\n"
             "Selected Statement: {statement}\n\n"
             "Function Inputs:\n"
             "{inputs}\n\n"
             "Please put your answer in the <ans></ans> tags"
             )
        self.pt_template_output = \
            ("Given the following {lang} code snippet and input of the code, "
             "what will be the output of the code given the input value?\n\n"
             "Code Snippet\n"
             "```{lang}\n"
             "{code}\n"
             "```\n\n"
             "Function Inputs:\n"
             "{inputs}\n\n"
             "Please put your answer in the <ans></ans> tags"
             )
        self.pt_template_input = \
            ("Given the following {lang} code snippet and output of the code, "
             "what will be the input of the code given the output value?\n\n"
             "Code Snippet\n"
             "```{lang}\n"
             "{code}\n"
             "```\n\n"
             "Function Output:\n"
             "{outputs}\n\n"
             "Please put your answer in the <ans></ans> tags"
             )
        self.pt_template_loop_iteration = \
            ("Given the following {lang} code snippet with function call showing the input of the code,\n\n"
             "Code Snippet\n"
             "```{lang}\n"
             "{code}\n"
             "```\n\n"
             "Question:\n"
             "{question}\n\n"
             "Please put your answer in the <ans></ans> tags"
             )
        self.pt_template_loop_body = \
            ("Given the following {lang} code snippet with function call showing the input of the code,\n\n"
             "Code Snippet\n"
             "```{lang}\n"
             "{code}\n"
             "```\n\n"
             "Question:\n"
             "{question}\n\n"
             "Please put your answer in the <ans></ans> tags"
             )
        self.pt_template_alias = \
            ("Given the following {lang} code snippet and its input parameters:\n\n"
            "You are given two pointer variables in the code:\n"
            "- Pointer A: {pointer_1} (line {line_1})\n"
            "- Pointer B: {pointer_2} (line {line_2})\n\n"
            "Determine if these pointers are aliases (reference the same memory address).\n"
            "Respond with:\n"
            "- \"Yes\" if they point to the same memory location\n"
            "- \"No\" if they point to different locations\n\n"
            "Code:\n"
            "```{lang}\n"
            "{code}\n"
            "```\n\n"
            "Function Input:\n"
            "{input}\n\n"
            "Question:\n"
            "Do {pointer_1} (line {line_1}) and {pointer_2} (line {line_2}) alias the same memory address?\n\n"
            "Provide your answer within <ans></ans> tags.")

    def generate_cot_steps(self, sample):
        st_type = sample['Statement Type']
        variables = sample['Variable Values Before Statement']
        statement = sample['Selected Statement']
        
        if st_type == "Assignment":
            return (
                f"1. Examine the assignment statement: '{statement}'\n"
                f"2. Current variable values: {variables}\n"
                "3. Evaluate the right-hand side expression using these values\n"
                "4. The result becomes the new value of the left-hand side variable"
            )
        elif st_type == "Branch":
            return (
                f"1. Examine the branch condition: '{statement}'\n"
                f"2. Current variable values: {variables}\n"
                "3. Evaluate the conditional expression using these values\n"
                "4. Determine if the condition is true or false\n"
                "5. This determines whether the branch will be taken"
            )
        elif st_type == "API":
            return (
                f"1. Examine the API call: '{statement}'\n"
                f"2. Current parameter values: {variables}\n"
                "3. Determine what this API/function does with these parameters\n"
                "4. Compute or predict the return value based on the function's logic"
            )
        else:
            return (
                f"1. Examine the statement: '{statement}'\n"
                f"2. Current context: {variables}\n"
                "3. Analyze how the statement transforms these values\n"
                "4. Determine the result after execution"
            )
            
    def demo2msg(self, demos):
        if self.args.prediction == "statement":
            if self.args.CoT == "no":
                msg = (
                    f"You will be given {self.args.language} code snippets with different types of statements "
                    "(assignment, branch, or function calls). For each, you'll see:\n"
                    "1. The complete code snippet\n"
                    "2. A highlighted statement\n"
                    "3. Variable values before that statement executes\n\n"
                    "Your task is to predict the value after the statement executes.\n\n"
                    f"Here are {self.args.shot} worked examples:\n\n"
                    "----------------------------------------\n"
                )
                
                for i, sample in enumerate(demos, 1):
                    if sample['Statement Type'] in ['Assignment', 'Constant Assignment', 'Arithmetic Assignment']:
                        template = getattr(self, "pt_template_Assignment")
                    else:
                        template = getattr(self, f"pt_template_{sample['Statement Type']}")
                    

                    example = template.format(
                        lang=sample['Programming Language'].lower(),
                        code=sample['Source Code'],
                        statement=sample['Selected Statement'],
                        variables=sample['Variable Values Before Statement']
                    )
                    
                    msg += f"EXAMPLE {i}:\n{example}\n"
                    msg += f"Correct Answer:<ans>{sample['Value After Statement Execution']}</ans>\n"
                
                msg += (
                    "\nNow, please solve the following new problem.\n\n"
                )
            else:
                msg = (
                    f"You will be given {self.args.language} code snippets with different types of statements "
                    "(assignment, branch, or function calls). For each, you'll see:\n"
                    "1. The complete code snippet\n"
                    "2. A highlighted statement\n"
                    "3. Variable values before that statement executes\n\n"
                    "Your task is to predict the value after the statement executes by thinking step by step.\n\n"
                    f"Here are {self.args.shot} worked examples with reasoning steps:\n\n"
                    "----------------------------------------\n"
                )
                
                for i, sample in enumerate(demos, 1):
                    if sample['Statement Type'] in ['Assignment', 'Constant Assignment', 'Arithmetic Assignment']:
                        template = getattr(self, "pt_template_Assignment")
                    else:
                        template = getattr(self, f"pt_template_{sample['Statement Type']}")
                    
                    example = template.format(
                        lang=sample['Programming Language'].lower(),
                        code=sample['Source Code'],
                        statement=sample['Selected Statement'],
                        variables=sample['Variable Values Before Statement']
                    )
                    
                    msg += f"EXAMPLE {i}:\n{example}\n"
                    
                    cot_steps = self.generate_cot_steps(sample)
                    msg += f"Let's think step by step:\n{cot_steps}\n"
                    msg += f"Therefore, the final answer is: <ans>{sample['Value After Statement Execution']}</ans>\n"
                    msg += "----------------------------------------\n"
                
                msg += (
                    "\nNow, please solve the following new problem. "
                    "Think through each step carefully and put your final answer in <ans></ans> tags.\n\n"
                )
            return msg
                
    def task2msg(self, task):
        pt =self.task2pt(task)
        msg = self._pt2msg(pt)
        return msg



    def task2pt(self, task: dict ):
        if self.args.prediction == "output":
            pt = self.pt_template_output.format(
                lang=self.args.language.lower(),
                code=task['code'],
                inputs=task['input'],
            )
        elif self.args.prediction == "input":
            pt = self.pt_template_input.format(
                lang=self.args.language.lower(),
                code=task['code'],
                outputs=task['output'],
            )
        elif self.args.prediction == "loop":
            if self.args.settings == "iteration":
                pt = self.pt_template_loop_iteration.format(
                    lang=self.args.language.lower(),
                    code=task['loop_code'],
                    question=task['question'],
                )
            elif self.args.settings == "body":
                pt = self.pt_template_loop_body.format(
                    lang=self.args.language.lower(),
                    code=task['loop_code'],
                    question=task['question'],
                )
        elif self.args.prediction == "alias":
            pt = self.pt_template_alias.format(
                lang=self.args.language.lower(),
                code=task['Source Code'],
                input= task['Function Input'],
                pointer_1 = task['Selected Pointer'],
                line_1 = task['Selected Statement'],
                pointer_2 = task['Compared Pointer'],
                line_2 = task['Compared Statement'], 
            )
        elif self.args.prediction == "block":
            pt = self.pt_template_block.format(
                lang=task['Programming Language'].lower(),
                code=task['Source Code'],
                statement=task['Selected Statement'],
                inputs=task['Function Input'],
            )
        elif self.args.prediction == "statement":
            if self.args.incontext == "same":
                demos = incontext_shots_with_same_statement(self.args, task)
            else:
                demos = self.demos 
            
            if task['Statement Type'] == "Branch":
                pt = self.pt_template_Branch.format(
                        lang=task['Programming Language'].lower(),
                        code=task['Source Code'],
                        statement=task['Selected Statement'],
                        variables=task['Variable Values Before Statement'],
                    )
                if self.args.shot == 0:
                    pt = pt
                else:
                    msg = self.demo2msg(demos)
                    pt = msg + pt
                    
                    
            elif task['Statement Type'] == "API":
                pt = self.pt_template_API.format(
                    lang=task['Programming Language'].lower(),
                    code=task['Source Code'],
                    statement=task['Selected Statement'],
                    variables=task['Variable Values Before Statement'],
                )
                if self.args.shot == 0:
                    pt = pt
                else:
                    msg = self.demo2msg(demos)
                    pt = msg + pt
            else:
                pt = self.pt_template_Assignment.format(
                    lang=task['Programming Language'].lower(),
                    code=task['Source Code'],
                    statement=task['Selected Statement'],
                    variables=task['Variable Values Before Statement'],
                )
                if self.args.shot == 0:
                    pt = pt
                else:
                    msg = self.demo2msg(demos)
                    pt = msg + pt
        return pt

    def extract_ans(self, prompt_str, llm_output_str):
        return self.extract_data(llm_output_str, tag_name="ans")

    def _pt2msg(self, pt):
        msg = [{"content": pt, "role": "user"}]
        return msg


    def msg2pt(self):
        raise NotImplementedError


class StatementPt2(StatementPt1):
    def __init__(self, name, demos):
        super().__init__(name, demos)

    def task2pt(self, task: dict ):
        ori_code = task['Source Code']
        selected_statement = task['Selected Statement']
        val_dict = task['Variable Values Before Statement']
        val_str = ','.join([f"{k}={val_dict[k]}" for k in val_dict])
        ori_code_st_list = ori_code.split("\n")
        new_code_st_list = []
        for ori_st in ori_code_st_list:
            if ori_st.strip() == selected_statement.strip():
                new_v_str = val_str.replace("\n", ",")
                new_code_st_list.append(ori_st + f'# {new_v_str}')
            else:
                new_code_st_list.append(ori_st)
        new_code = "\n".join(new_code_st_list)
        if task['Statement Type'] == "Branch":
            pt = self.pt_template_branch.format(
                lang=task['Programming Language'].lower(),
                code=new_code,
                statement=task['Selected Statement'],
                variables=task['Variable Values Before Statement'],
            )
        elif task['Statement Type'] == "API":
            pt = self.pt_template_api.format(
                lang=task['Programming Language'].lower(),
                code=new_code,
                statement=task['Selected Statement'],
                variables=task['Variable Values Before Statement'],
            )
        else:
            pt = self.pt_template_assignment.format(
                lang=task['Programming Language'].lower(),
                code=new_code,
                statement=task['Selected Statement'],
                variables=task['Variable Values Before Statement'],
            )
        return pt

class StatementPt3(StatementPt1):
    def __init__(self, name, demos):
        super().__init__(name, demos)

    def task2pt(self, task: dict):
        ori_code = task['Source Code']
        new_code = ori_code
        pt = self.pt_template.format(
            lang=task['Programming Language'].lower(),
            code=new_code,
            statement=task['Selected Statement'],
            variables=task['Variable Values Before Statement'],
        )
        return pt