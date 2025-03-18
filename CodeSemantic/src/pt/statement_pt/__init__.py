


from ..abst_pt import AbstPt


class StatementPt1(AbstPt):
    def __init__(self, name, demos):
        super().__init__(name, demos)
        self.pt_template = \
            ("Given the following f{lang} code snippet and the selected statement, "
             "the local variable values before the statements are shown as follows, "
             "what will be the value after the selected statement after executing the selected statement.\n\n"
             "Code Snippet\n"
             "```{lang}\n"
             "{code}\n"
             "```\n\n"
             "Selected Statement: {statement}\n\n"
             "Local Variables:\n"
             "{variables}\n\n"
             "Please put your answer in the <ans></ans> tags"
             )


    def task2msg(self, task):
        pt =self.task2pt(task)
        msg = self._pt2msg(pt)
        return msg



    def task2pt(self, task: dict ):
        pt = self.pt_template.format(
            lang=task['Programming Language'].lower(),
            code=task['Source Code'],
            statement=task['Selected Statement'],
            variables=task['Variable Values Before Statement'],
        )
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
        pt = self.pt_template.format(
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