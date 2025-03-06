import streamlit as st
import pdfplumber
from langchain_ollama import OllamaLLM
from fpdf import FPDF
import pandas as pd
from docx import Document
import base64
from my_api import get_response
import subprocess
import tempfile

####ONE-------------------------------------------------------------------------import done

class PDFWithBorder(FPDF):
    def header(self):
        """Add a border to every page."""
        self.set_draw_color(0, 0, 0)  # Black color
        self.set_line_width(0.5)
        self.rect(5, 5, 200, 287)  # Rect(x, y, width, height)

    def convert_brd_to_pdf(brd_text, output_pdf_path):
        # Create a PDF object
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        
        # Split the BRD text into lines
        lines = brd_text.split("\n")
        
        # Process each line
        for line in lines:
            words = line.split(" ")
            
            # Loop through words and check for * symbols indicating bold text
            for i, word in enumerate(words):
                if word.startswith("") and word.endswith(""):
                    # Apply bold style
                    words[i] = word[1:-1]  # Remove the '*' symbols
                    pdf.set_font("Arial", style='B', size=12)
                else:
                    pdf.set_font("Arial", size=12)

            # Join the words and add them to the PDF
            pdf.cell(200, 10, txt=" ".join(words), ln=True)

        # Output the PDF
        pdf.output(output_pdf_path)

    # Example BRD text (with * symbols indicating bold text)
    brd_text = """
    """

    # Specify the output PDF file path
    output_pdf_path = "output_brd.pdf"

    # Convert BRD text to PDF
    #convert_brd_to_pdf(brd_text, output_pdf_path)

    print(f"PDF saved as {output_pdf_path}")
    
####TWO-------------------------------------------------------------------------brd conversion done

# Backend Module
class BRDProcessor:
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.2")

    def extract_text(self, uploaded_file, file_type):
        if file_type == "pdf":
            return self._extract_text_from_pdf(uploaded_file)#three
        elif file_type == "docx":
            return self._extract_text_from_docx(uploaded_file)#three
        elif file_type == "txt":
            return self._extract_text_from_txt(uploaded_file)#three
        return None

####THREE-------------------------------------------------------------------------file type finding...

    def _extract_text_from_pdf(self, pdf_file):
        with pdfplumber.open(pdf_file) as pdf:
            return ''.join([self._sanitize_text(page.extract_text() + '\n') for page in pdf.pages])

    def _extract_text_from_docx(self, docx_file):
        doc = Document(docx_file)
        return self._sanitize_text('\n'.join([paragraph.text for paragraph in doc.paragraphs]))

    def _extract_text_from_txt(self, txt_file):
        return self._sanitize_text(txt_file.read().decode('utf-8'))

    def _sanitize_text(self, text):
        """Remove unsupported characters and sanitize text."""
        return text.encode('utf-8', 'ignore').decode('utf-8')

####FOUR-------------------------------------------------------------------------text extraction

    def generate_requirements_table(self, brd_content):
        refined_content = self._preprocess_brd_content(brd_content)#5
        prompt = (
            f"Analyze the following BRD content and identify all the possible Use Cases that can happen. "
            f"Each Use Case should have a unique ID, type ('Functional' or 'Non-Functional'), and description. "
            f"Output in the format: | ID | Requirement Type | Description |\n\n"
            f"ID should be of the format 'F-m','NF-n','TR-o'. Where m, n, o denote the next number for indexing. "
            f"Ensure the indexing for functional, non-functional, and training is independently calculated.\n"
            f"Content:\n{refined_content}\n\n"
        )
        table_text = self.llm.invoke(prompt)
        return self._parse_requirements_table(table_text)#six
####FIVE-------------------------------------------------------------------------the requirement table is made..(not put into""table "tho)


    def _preprocess_brd_content(self, brd_content):
        prompt = (
            f"Analyze the following Business Requirements Document (BRD) content. "
            f"Extract the sections that describe the requirements, including functional and non-functional requirements, "
            f"and return only those sections in a concise manner.\n\n"
            f"BRD Content:\n{brd_content}\n\n"
            f"Return only the relevant parts of the document related to the requirements."
        )
    
        # Use the LLM to process and extract relevant content
        refined_content = self.llm.invoke(prompt)
        return refined_content.strip()
        
####SIX-------------------------------------------------------------------------taking the good stuff done

    def _parse_requirements_table(self, table_text):
        if not table_text.strip():
            return pd.DataFrame(columns=["ID", "Requirement Type", "Description"])

        try:
            rows = table_text.strip().split("\n")[1:]  # Skip the header row
            valid_rows = []
            for row in rows:
                if "|" in row:
                    columns = row.split("|")[1:-1]
                    if len(columns) == 3:  # Ensure it has exactly 3 columns
                        valid_rows.append(columns)

            return pd.DataFrame(valid_rows, columns=["ID", "Requirement Type", "Description"])
        except Exception as e:
            st.error(f"Error parsing requirements table: {e}")
            return pd.DataFrame(columns=["ID", "Requirement Type", "Description"])
####SEVEN------------------------------------------------------------------------- the table is a more or less a table now(as df)
####################-----------CHECK-----------##############################
    def create_user_story(self, description=None, brd_content=None, selected_id=None):
        if description:
            content_to_use = description
            prompt = (
                f"Using the provided description, generate a user story for requirement ID '{selected_id}' (if provided). "
                f"Structure the user story with sections: Actors, Preconditions, Main Flow, Postconditions, and Exceptions.\n\n"
                f"Description:\n{content_to_use}\n"
                )
        elif brd_content:
            content_to_use = self._preprocess_brd_content(brd_content)#five
            prompt = (
                f"Using the provided BRD content, generate a general user story. "
                f"Structure the user story with sections: Actors, Preconditions, Main Flow, Postconditions, and Exceptions."
                f"no need for use case diagram generation\n\n"
                f"Content:\n{content_to_use}\n"
                )
        else:
            return "Error: No content provided to generate a user story."

        return self._sanitize_text(self.llm.invoke(prompt))#three
####EIGHT-------------------------------------------------------------------------made userstories from either the desc or the brd content(mostly the desc)done

    def generate_use_case(self, user_story):
        prompt = (
            f"Based on the following user story, generate a use case with detailed sections like Actors, Preconditions, "
            f"Main Flow, Postconditions, and Exceptions. The use case should be dynamic, not using a hardcoded template.\n\n"
            f"User Story:\n{user_story}\n"
        )
        return self._sanitize_text(self.llm.invoke(prompt))#three
    


    def generate_test_case_scenario(self, use_case):
        prompt = (
            f"Based on the following use case, generate a test case scenario that outlines the conditions for testing "
            f"the functionality described in the use case. The test case scenario should include the purpose, inputs, "
            f"expected outputs, and any relevant setup or conditions.\n\n"
            f"Use Case:\n{use_case}\n"
        )
        return self._sanitize_text(self.llm.invoke(prompt))#3

    def generate_test_case(self, test_case_scenario):
        prompt = (
            f"Based on the following test case scenario, generate a detailed test steps, expected results, "
            f"preconditions,test data,expected results,pass/fail criteria."
            f"type of test if possible\n\n"
            f"Test Case Scenario:\n{test_case_scenario}\n"
        )
        return self._sanitize_text(self.llm.invoke(prompt))#3
####NINE-------------------------------------------------------------------------project basically over here all content is generated


    def generate_use_test_case_pdf(self, use_case, test_case,final_code,file_name):
        def remove_asterisks(text):
            return text.replace('*', '')  # Remove * symbols

        pdf = PDFWithBorder()#1one

        # Add the Use Case Document page
        pdf.add_page()
        pdf.set_font("Arial", 'B', size=16)
        pdf.cell(0, 10, "Use Case Document", ln=True, align="C")
        pdf.ln(10)

        # Add the Use Case content
        pdf.set_font("Arial", size=12)
        use_case = remove_asterisks(use_case)
        pdf.multi_cell(190, 10, use_case)

        # Add the Test Case Document page
        pdf.add_page()
        pdf.set_font("Arial", 'B', size=16)
        pdf.cell(0, 10, "Test Case Document", ln=True, align="C")
        pdf.ln(10)

        # Add the Test Case content
        pdf.set_font("Arial", size=12)
        test_case = remove_asterisks(test_case)
        pdf.multi_cell(190, 10, test_case)

        # Add the tested Code Document page
        pdf.add_page()
        pdf.set_font("Arial", 'B', size=16)
        pdf.cell(0, 10, "Final Code", ln=True, align="C")
        pdf.ln(10)

        # Add the tested Code content
        pdf.set_font("Courier", size=10)
        final_code = remove_asterisks(final_code)
        pdf.multi_cell(190, 5, final_code)


        # Save the PDF
        pdf.output(file_name)######################################-----------CHECK
####TEN-------------------------------------------------------------------------pdf is made..tuff

    def generate_code_from_use_case(self, use_case):
        prompt = (
            f"Based on the following use case, generate code that implements the described functionality or demonstrates "
            f"its behavior. Include appropriate comments and ensure it adheres to standard coding practices. "
            f"The code should focus on the functional aspects outlined in the use case. And also the code should cover all the requirements and edge cases of the given use case.Can you not generate explanations in the generated code.Also the generated code should not contain any error.The generated code should be error free.Also the generated code should provide output. It is compulsory to omit code block markers like '''python and ''' from the output.\n\n"
            f"Use Case:\n{use_case}\n"
        )
        return self._sanitize_text(get_response(prompt))#3
##########bakki nokkanam######


    def validate_code_with_llm(self, code, test_cases, use_case):
#Validates the generated code using an LLM and regenerates if necessary.
        max_attempts = 5  # Maximum retries to get correct code
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            st.write(f"### 🛠 Attempt {attempt}: Testing Generated Code with LLM...")
        
        # Prepare the prompt for LLM
            prompt = f"""
            You are a code execution assistant. Given the following Python code, execute it and compare the output against expected test cases. 
        
            *Code:*
            python
            {code}
            
            *Test Cases (Expected Outputs):*
            {test_cases}
        
            Return 'PASSED' if all test cases are met, otherwise return 'FAILED'.
            """
        
        # Get LLM response
            response = get_response(prompt).strip()
        
        # Check LLM's verdict
            if "PASSED" in response:
                st.success("✅ Code passed all test cases!")
                return code  # Return the successful code
        
            st.error("❌ Code failed some test cases. Regenerating...")
        
        # Generate new code if test cases failed
            code = self.generate_code_from_use_case(use_case)
    
        st.error("❌ Maximum attempts reached. Code couldn't pass all test cases.")
        return code  # Return the best attempt


        

####ELEVEN-------------------------------------------------------------------------code is made,integration of code is a status fail and testing part is also not fully made

############-----------------------backend done-----------------------############
# UI Module: Display and Download
class UseCaseApp:
    def __init__(self):
        self.processor = BRDProcessor()


    def run(self):
        st.set_page_config(page_title="Generation of Use Case, Test Case, and Code", layout="wide")

        # Sidebar instructions
        self.sidebar_instructions()

        # Background image setup
        image_base64 = self.convert_image_to_base64(r"C:\Users\elz00\Desktop\project\2.png")
        st.markdown(
            f"""
            <style>
                .stApp {{
                    background-image: url("data:image/jpeg;base64,{image_base64}");
                    background-repeat: no-repeat;
                    background-size: cover;
                    background-attachment: fixed;
                }}
                .title {{
                    color: darkred;
                    text-align: center;
                    font-weight: bold;
                }}
            </style>
            """, unsafe_allow_html=True
        )

        st.markdown('<h1 class="title">Generation of Use Case, Test Case, and Code</h1>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader("📎 Upload a BRD file:", type=["pdf", "docx", "txt"])
        if uploaded_file:
            file_type = uploaded_file.name.split('.')[-1]
            extracted_text = self.processor.extract_text(uploaded_file, file_type)#TWO

            if extracted_text:
                st.success("BRD file uploaded and text extracted successfully.")

                if "requirements_df" not in st.session_state:
                    st.session_state.requirements_df = self.processor.generate_requirements_table(extracted_text)#four

                st.write("### Requirements Table:")
                st.dataframe(st.session_state.requirements_df, use_container_width=True, height=500)

                selected_id = st.selectbox(
                    "Select a Requirement ID to generate a specific user story:",
                    options=["Entire"] + st.session_state.requirements_df["ID"].tolist(),
                )
                selected_id = None if selected_id == "Entire" else selected_id

                # Extract description for the selected ID
                description = None
                if selected_id:
                    description = st.session_state.requirements_df.loc[
                        st.session_state.requirements_df["ID"] == selected_id, "Description"
                    ].values[0]
                    st.write("### Selected Description:")
                    st.write(description)

                if st.button("Generate User Story, Use Case, Test Case, and Code"):
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.spinner("Generating user story..."):
                            user_story = self.processor.create_user_story(description=description, brd_content=extracted_text, selected_id=selected_id)
                            with st.expander("USER STORY"):
                                st.subheader("Generated User Story")
                                st.write(user_story)
                    with col2:
                        with st.spinner("Generating use case..."):
                            use_case = self.processor.generate_use_case(user_story)#eight
                            with st.expander("USE CASE"):
                                st.subheader("Generated Use Case")
                                st.write(use_case)
                            
                    col3, col4 = st.columns(2)
                    with col3:
                        with st.spinner("Generating test case scenario..."):
                            test_case_scenario = self.processor.generate_test_case_scenario(use_case)#eight
                            with st.expander("TEST CASE SCENARIO"):
                                st.subheader("Generated Test Case Scenario")
                                st.write(test_case_scenario)

                    with col4:
                        with st.spinner("Generating test case..."):
                            test_case = self.processor.generate_test_case(test_case_scenario)#eight
                            with st.expander("TEST CASE"):
                                st.subheader("Generated Test Case")
                                st.write(test_case)

                    with st.spinner("Generating initial code..."):
                        generated_code = self.processor.generate_code_from_use_case(use_case)#ten
                        with st.expander("GENERATED CODE"):
                            st.subheader("Generated Code")
                            st.code(generated_code)
                    
####TWELVE-------------------------------------------------------------------------getting all the details from user done


                    final_code = self.processor.validate_code_with_llm(generated_code, test_case, use_case)

                    with st.expander("✅ Final Corrected Code"):
                        st.code(final_code)
                        
                    # Generate the PDF with the new content
                    pdf_name = "use_case_test_case_code_document.pdf"
                    self.processor.generate_use_test_case_pdf(use_case, test_case, final_code, pdf_name)#nine
                    
                    with open(pdf_name, "rb") as pdf_file:
                        st.download_button("🔄 Download PDF", pdf_file, file_name=pdf_name)
            else:
                st.error("Failed to extract text from the uploaded file.")
####THIRTEEN-------------------------------------------------------------------------making ends meet... everything is here..(ui)

    # Sidebar instructions
    def sidebar_instructions(self):
        st.sidebar.title("Instructions")
        st.sidebar.info(
            """
            1. Upload a BRD in PDF, DOCX, or TXT format.
            2. Select a specific requirement ID (optional).
            3. Generate the user story, use case, test case, and corresponding code.
            4. Download the generated content as a PDF.
            """
        )

    def display_use_case(self, use_case):
        st.subheader("Generated Use Case")
        st.write(use_case)

    def convert_image_to_base64(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            st.error("Background image not found. Please check the file path.")
            return ""

        
############-----------------------frontend done-----------------------############
#everything is over....
# Main entry point
if __name__ == "__main__":
    app = UseCaseApp()
    app.run()
