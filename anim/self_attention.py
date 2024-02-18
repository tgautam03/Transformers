import numpy as np
from manim import *
import pickle

from utils import remove_invisible_chars

class Dataset(Scene):
    def construct(self):
        # Setting background colour
        self.camera.background_color = "#282a36"
        # Self Attention box
        title = Text("Self Attention")
        box = SurroundingRectangle(title, color=WHITE, buff=MED_LARGE_BUFF)
        self.play(Write(title))
        self.wait(1)
        self.play(Create(box))
        self.wait(1)
        # Demo Input and output sequences
        in_seq = Matrix([[1, 2, 3, 4, 5]]).scale(0.6).next_to(box, LEFT)
        out_seq = Matrix([[5, 4, 3, 2, 1]]).scale(0.6).next_to(box, RIGHT)
        self.play(Write(in_seq))
        self.wait(1)
        self.play(FadeOut(in_seq, target_position=title))
        self.play(FadeIn(out_seq, target_position=title))
        self.wait(1)
        self.play(FadeOut(title, out_seq))
        self.play(box.animate.scale(8), run_time=2)
        self.wait(1)

        # IMDB dataset
        title = Text("IMDB Dataset").to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        with open ('review1.pkl', 'rb') as fp:
            review1 = pickle.load(fp)
        review1 = Text(review1).scale(0.3).next_to(title, DOWN)
        label = Text("Label: 1").scale(0.5).next_to(review1, DOWN)
        self.play(Write(review1))
        self.wait(1)
        self.play(Write(label))
        self.wait(1)
        comment = Text("There are 20000 more reviews just like this!").scale(0.5).next_to(label, DOWN).shift(3*DOWN)
        self.play(Write(comment))
        self.wait(1)
        self.play(FadeOut(label))
        with open ('review2.pkl', 'rb') as fp:
            review2 = pickle.load(fp)
        review2 = Text(review2).scale(0.3).next_to(review1, DOWN)
        self.play(Write(review2))
        with open ('review3.pkl', 'rb') as fp:
            review3 = pickle.load(fp)
        review3 = Text(review3).scale(0.3).next_to(review2, DOWN)
        self.play(Write(review3))
        vdots = Tex(r"$\vdots$").scale(0.5).next_to(review3, DOWN)
        with open ('review4.pkl', 'rb') as fp:
            review4 = pickle.load(fp)
        review4 = Text(review4).scale(0.3).next_to(vdots, DOWN)
        self.play(ReplacementTransform(comment, VGroup(vdots, review4)))
        self.wait(1)

        # Tokenization and Dictionary building
        with open ('vocab2.pkl', 'rb') as fp:
            vocab2 = pickle.load(fp)
        vocab2 = Text(vocab2).scale(0.3).to_edge(DOWN)
        vdots_ = Tex(r"\vdots").scale(0.5).next_to(vocab2, UP)
        with open ('vocab1.pkl', 'rb') as fp:
            vocab1 = pickle.load(fp)
        vocab1 = Text(vocab1).scale(0.3).next_to(vdots_, UP)
        vocab_box = SurroundingRectangle(VGroup(vocab1, vdots_, vocab2), color=WHITE, buff=MED_SMALL_BUFF)
        self.play(Create(vocab_box))
        self.wait(1)
        self.play(ReplacementTransform(VGroup(review1, review2, review3, vdots, review4), VGroup(vocab1, vdots_, vocab2)))
        self.wait(1)

        with open ('num_vocab2.pkl', 'rb') as fp:
            num_vocab2 = pickle.load(fp)
        num_vocab2 = Text(num_vocab2).scale(0.3).to_edge(DOWN)
        num_vdots_ = Tex(r"\vdots").scale(0.5).next_to(num_vocab2, UP)
        with open ('num_vocab1.pkl', 'rb') as fp:
            num_vocab1 = pickle.load(fp)
        num_vocab1 = Text(num_vocab1).scale(0.3).next_to(num_vdots_, UP)
        self.wait(1)
        self.play(ReplacementTransform(VGroup(vocab1, vdots_, vocab2), VGroup(num_vocab1, num_vdots_, num_vocab2)))
        self.wait(1)

        with open ('review1.pkl', 'rb') as fp:
            review1 = pickle.load(fp)
        review1 = Text(review1).scale(0.3).next_to(title, DOWN)
        self.play(Write(review1))
        with open ('review2.pkl', 'rb') as fp:
            review2 = pickle.load(fp)
        review2 = Text(review2).scale(0.3).next_to(review1, DOWN)
        self.play(Write(review2))
        with open ('review3.pkl', 'rb') as fp:
            review3 = pickle.load(fp)
        review3 = Text(review3).scale(0.3).next_to(review2, DOWN)
        self.play(Write(review3))
        vdots = Tex(r"$\vdots$").scale(0.5).next_to(review3, DOWN)
        with open ('review4.pkl', 'rb') as fp:
            review4 = pickle.load(fp)
        review4 = Text(review4).scale(0.3).next_to(vdots, DOWN)  
        self.play(Write(vdots), Write(review4))  
        
        with open ('num_review1.pkl', 'rb') as fp:
            num_review1 = pickle.load(fp)
        num_review1 = Text(num_review1).scale(0.3).move_to(review1, DOWN)
        self.play(ReplacementTransform(review1, num_review1))
        with open ('num_review2.pkl', 'rb') as fp:
            num_review2 = pickle.load(fp)
        num_review2 = Text(num_review2).scale(0.3).move_to(review2, DOWN)
        self.play(ReplacementTransform(review2, num_review2))
        with open ('num_review3.pkl', 'rb') as fp:
            num_review3 = pickle.load(fp)
        num_review3 = Text(num_review3).scale(0.3).move_to(review3, DOWN)
        self.play(ReplacementTransform(review3, num_review3))
        with open ('num_review4.pkl', 'rb') as fp:
            num_review4 = pickle.load(fp)
        num_review4 = Text(num_review4).scale(0.3).move_to(review4, DOWN)  
        self.play(ReplacementTransform(review4, num_review4))  
        self.wait(1)

        self.play(FadeOut(vocab_box, num_vocab1, num_vdots_, num_vocab2))
        comment = Text("Different sentences have different lengths...").scale(0.5).next_to(label, DOWN).shift(3*DOWN)
        self.play(Write(comment))
        self.wait(1)
        comment1 = Text("Padding to make them of equal lengths!").scale(0.5).next_to(label, DOWN).shift(3*DOWN)

        with open ('padded_num_review1.pkl', 'rb') as fp:
            padded_num_review1 = pickle.load(fp)
        padded_num_review1 = Text(padded_num_review1).scale(0.3).move_to(review1, DOWN)
        self.play(ReplacementTransform(num_review1, padded_num_review1), ReplacementTransform(comment, comment1))
        with open ('padded_num_review2.pkl', 'rb') as fp:
            padded_num_review2 = pickle.load(fp)
        padded_num_review2 = Text(padded_num_review2).scale(0.3).move_to(review2, DOWN)
        self.play(ReplacementTransform(num_review2, padded_num_review2))
        with open ('padded_num_review3.pkl', 'rb') as fp:
            padded_num_review3 = pickle.load(fp)
        padded_num_review3 = Text(padded_num_review3).scale(0.3).move_to(review3, DOWN)
        self.play(ReplacementTransform(num_review3, padded_num_review3))
        with open ('padded_num_review4.pkl', 'rb') as fp:
            padded_num_review4 = pickle.load(fp)
        padded_num_review4 = Text(padded_num_review4).scale(0.3).move_to(review4, DOWN)  
        self.play(ReplacementTransform(num_review4, padded_num_review4))  
        self.wait(1)

        self.play(FadeOut(comment1, padded_num_review4, padded_num_review3, padded_num_review2, vdots, padded_num_review1[10:]))
        ex_review = VGroup(Text("14"),
                           Text("19"),
                           Text("10"),
                           Text("59"),
                           Text("57")).scale(0.5).arrange(DOWN, aligned_edge=LEFT)
        self.play(ReplacementTransform(padded_num_review1[:10], ex_review))
        self.wait(1)

        emb_token = np.load("ex_review_emb.npy")
        ex_review_emb = Matrix(np.round(emb_token, 2), v_buff=1.5, h_buff=1.5).scale(0.35).next_to(title, DOWN).shift(0.75*DOWN)
        self.play(ReplacementTransform(ex_review, ex_review_emb))
        self.wait(1)

        emb_code = Code(code="""
                            token_embedding = torch.nn.Embedding(embedding_dim=7, num_embeddings=len(vocab))
                            emb_token = token_embedding(ex_review)
                            """, language="Python", font="Monospace", insert_line_no=False,
                            style="dracula", line_spacing=1).scale(0.5).next_to(ex_review_emb, DOWN).shift(DOWN)
        emb_code.code = remove_invisible_chars(emb_code.code)
        self.play(Create(emb_code))
        self.wait(1)
        box1 = SurroundingRectangle(emb_code.code[0][35:50], buff=0.05)
        t1 = Text("Vector Length").scale(0.3).next_to(box1, UP)
        box2 = SurroundingRectangle(emb_code.code[0][51:-1], buff=0.05)
        t2 = Text("Vocab Size").scale(0.3).next_to(box2, UP)
        self.play(Create(box1), Write(t1))
        self.wait(1)
        self.play(Create(box2), Write(t2))
        self.wait(1)
        self.play(FadeOut(emb_code, box1, box2, t1, t2))

        mat_dim = Tex(r"1", r"$\times$", r"5", r"$\times$", r"7").scale(2).next_to(ex_review_emb, DOWN).shift(0.5*DOWN)
        t1 = Text("Total Reviews").scale(0.3).next_to(mat_dim[0], DOWN)
        t2 = Text("Total Words").scale(0.3).next_to(mat_dim[2], UP)
        t3 = Text("Embedding Size").scale(0.3).next_to(mat_dim[-1], DOWN)
        self.play(Write(mat_dim))
        self.wait(1)
        self.play(Write(t1))
        self.wait(1)
        self.play(Write(t2))
        self.wait(1)
        self.play(Write(t3))
        self.wait(1)
        self.play(FadeOut(title))
        self.wait(1)

        return super().construct()
    
class SelfAttentionOp(Scene):
    def construct(self):
        # Setting Scene
        title = Text("Self Attention").to_edge(UP)
        emb_token = np.load("ex_review_emb.npy")
        ex_review_emb = Matrix(np.round(emb_token, 2), v_buff=1.5, h_buff=1.5).set_row_colors(RED_A, GREEN_A, BLUE_A, ORANGE, YELLOW).scale(0.4).next_to(title, DOWN).shift(0.75*DOWN)
        mat_dim = Tex(r"1", r"$\times$", r"5", r"$\times$", r"7").scale(2).next_to(ex_review_emb, DOWN).shift(0.5*DOWN)
        t1 = Text("Total Reviews").scale(0.3).next_to(mat_dim[0], DOWN)
        t2 = Text("Total Words").scale(0.3).next_to(mat_dim[2], UP)
        t3 = Text("Embedding Size").scale(0.3).next_to(mat_dim[-1], DOWN)
        self.add(title, ex_review_emb, mat_dim, t1, t2, t3)
        self.wait(1)

        # Self Attention
        self.play(FadeOut(mat_dim, t1, t2, t3))
        self.wait(1)
        word1 = Text("This", color=RED_A).scale(0.3).next_to(ex_review_emb.get_rows()[0], LEFT).shift(0.5*LEFT)
        word2 = Text("movie", color=GREEN_A).scale(0.3).next_to(ex_review_emb.get_rows()[1], LEFT).shift(0.35*LEFT)
        word3 = Text("is", color=BLUE_A).scale(0.3).next_to(ex_review_emb.get_rows()[2], LEFT).shift(0.5*LEFT)
        word4 = Text("very", color=ORANGE).scale(0.3).next_to(ex_review_emb.get_rows()[3], LEFT).shift(0.4*LEFT)
        word5 = Text("good", color=YELLOW).scale(0.3).next_to(ex_review_emb.get_rows()[4], LEFT).shift(0.5*LEFT)
        self.play(Write(word1), Write(word2), Write(word3), Write(word4), Write(word5))
        self.wait(1)
        self.play(FadeOut(VGroup(word1, word2, word3, word4, word5)))
        self.wait(1)
        W_mat = Matrix([["w_{00}", "w_{01}", "w_{02}", "w_{03}", "w_{04}"],
                        ["w_{10}", "w_{11}", "w_{12}", "w_{13}", "w_{14}"],
                        ["w_{20}", "w_{21}", "w_{22}", "w_{23}", "w_{24}"],
                        ["w_{30}", "w_{31}", "w_{32}", "w_{33}", "w_{34}"],
                        ["w_{40}", "w_{41}", "w_{42}", "w_{43}", "w_{44}"]], v_buff=1.5, h_buff=1.5).scale(0.4).next_to(title, DOWN).shift(0.75*DOWN).to_edge(LEFT)
        self.play(Write(W_mat), ex_review_emb.animate.next_to(W_mat))
        eq_sign = MathTex("=").next_to(ex_review_emb)
        self.wait(1)
        y_mat = Matrix([["y_{00}", "y_{01}", "y_{02}", "y_{03}", "y_{04}", "y_{05}", "y_{06}"],
                        ["y_{10}", "y_{11}", "y_{12}", "y_{13}", "y_{14}", "y_{15}", "y_{16}"],
                        ["y_{20}", "y_{21}", "y_{22}", "y_{23}", "y_{24}", "y_{25}", "y_{26}"],
                        ["y_{30}", "y_{31}", "y_{32}", "y_{33}", "y_{34}", "y_{35}", "y_{36}"],
                        ["y_{40}", "y_{41}", "y_{42}", "y_{43}", "y_{44}", "y_{45}", "y_{46}"]], v_buff=1.5, h_buff=1.5).set_row_colors(RED_A, GREEN_A, BLUE_A, ORANGE, YELLOW).scale(0.4).next_to(eq_sign, RIGHT)
        label1 = Text("Weight Matrix").scale(0.3).next_to(W_mat, UP)
        label2 = Text("Input Sequence").scale(0.3).next_to(ex_review_emb, UP)
        label3 = Text("Output Sequence").scale(0.3).next_to(y_mat, UP)
        self.play(Write(label1), Write(label2))
        self.wait(1)
        self.play(Create(eq_sign), Create(y_mat), Write(label3))
        self.wait(1)
        out_eq = MathTex(r"Y=W \cdot X").scale(0.75).to_edge(DOWN)
        self.play(Write(out_eq))
        self.wait(1)

        self.play(FadeOut(title))
        title = Text("Weight Matrix").to_edge(UP)
        self.play(Write(title), FadeOut(y_mat, label1, label2, label3))
        self.wait(1)
        self.play(eq_sign.animate.next_to(W_mat, RIGHT))
        self.play(ex_review_emb.animate.next_to(eq_sign, RIGHT))
        ex_review_emb_T = Matrix(np.round(emb_token.T, 2), v_buff=1.5, h_buff=1.5).set_column_colors(RED_A, GREEN_A, BLUE_A, ORANGE, YELLOW).scale(0.4).next_to(ex_review_emb, RIGHT)
        self.play(Write(ex_review_emb_T))
        self.wait(1)

        W_eq2 = Tex(r"$W$", r"$=$", r"Softmax($W'$, axis=1)").scale(0.75).next_to(out_eq, UP).shift(0.25*UP)
        W_eq1 = Tex(r"$W'$", r"$=$", r"$\frac{(X \cdot X^T)}{\sqrt{7}}$").scale(0.75).next_to(W_eq2, UP).shift(0.25*UP)
        self.play(Write(W_eq1[0:2]), Write(W_eq1[2][:6]))
        self.wait(1)
        self.play(Write(W_eq1[2][6:]))
        self.wait(1)
        self.play(FadeOut(eq_sign, ex_review_emb_T, ex_review_emb), W_mat.animate.next_to(W_eq1, UP).shift(0.25*UP))
        self.wait(1)
        W_mat_ = Matrix([["\\frac{e^{w_{00}}}{\\sum_j e^{w_{0j}}}", "\\frac{e^{w_{01}}}{\\sum_j e^{w_{0j}}}", "\\frac{e^{w_{02}}}{\\sum_j e^{w_{0j}}}", "\\frac{e^{w_{03}}}{\\sum_j e^{w_{0j}}}", "\\frac{e^{w_{04}}}{\\sum_j e^{w_{0j}}}"],
                        ["\\frac{e^{w_{10}}}{\\sum_j e^{w_{1j}}}", "\\frac{e^{w_{11}}}{\\sum_j e^{w_{1j}}}", "\\frac{e^{w_{12}}}{\\sum_j e^{w_{1j}}}", "\\frac{e^{w_{13}}}{\\sum_j e^{w_{1j}}}", "\\frac{e^{w_{14}}}{\\sum_j e^{w_{1j}}}"],
                        ["\\frac{e^{w_{20}}}{\\sum_j e^{w_{2j}}}", "\\frac{e^{w_{21}}}{\\sum_j e^{w_{2j}}}", "\\frac{e^{w_{22}}}{\\sum_j e^{w_{2j}}}", "\\frac{e^{w_{23}}}{\\sum_j e^{w_{2j}}}", "\\frac{e^{w_{24}}}{\\sum_j e^{w_{2j}}}"],
                        ["\\frac{e^{w_{30}}}{\\sum_j e^{w_{3j}}}", "\\frac{e^{w_{31}}}{\\sum_j e^{w_{3j}}}", "\\frac{e^{w_{32}}}{\\sum_j e^{w_{3j}}}", "\\frac{e^{w_{33}}}{\\sum_j e^{w_{3j}}}", "\\frac{e^{w_{34}}}{\\sum_j e^{w_{3j}}}"],
                        ["\\frac{e^{w_{40}}}{\\sum_j e^{w_{4j}}}", "\\frac{e^{w_{41}}}{\\sum_j e^{w_{4j}}}", "\\frac{e^{w_{42}}}{\\sum_j e^{w_{4j}}}", "\\frac{e^{w_{43}}}{\\sum_j e^{w_{4j}}}", "\\frac{e^{w_{44}}}{\\sum_j e^{w_{4j}}}"]], v_buff=1.5, h_buff=2.5).scale(0.45).move_to(W_mat)
        self.play(ReplacementTransform(W_mat, W_mat_), Write(W_eq2))
        self.wait(1)

        self.play(FadeOut(W_mat_, title))
        self.wait(1)
        emb_token = np.load("ex_review_emb.npy")
        ex_review_emb = Matrix(np.round(emb_token, 2), v_buff=1.5, h_buff=1.5).set_row_colors(RED_A, GREEN_A, BLUE_A, ORANGE, YELLOW).scale(0.4).next_to(title, DOWN).to_edge(LEFT).shift(0.75*DOWN+0.4*RIGHT)
        word1 = Text("This", color=RED_A).scale(0.3).next_to(ex_review_emb.get_rows()[0], LEFT).shift(0.3*LEFT)
        word2 = Text("movie", color=GREEN_A).scale(0.3).next_to(ex_review_emb.get_rows()[1], LEFT).shift(0.1*LEFT)
        word3 = Text("is", color=BLUE_A).scale(0.3).next_to(ex_review_emb.get_rows()[2], LEFT).shift(0.3*LEFT)
        word4 = Text("very", color=ORANGE).scale(0.3).next_to(ex_review_emb.get_rows()[3], LEFT).shift(0.2*LEFT)
        word5 = Text("good", color=YELLOW).scale(0.3).next_to(ex_review_emb.get_rows()[4], LEFT).shift(0.3*LEFT)
        mat_dim = Tex(r"1", r"$\times$", r"5", r"$\times$", r"7").scale(0.5).next_to(ex_review_emb, DOWN)
        self.play(Write(word1), Write(word2), Write(word3), Write(word4), Write(word5), Create(ex_review_emb), Write(mat_dim))
        self.wait(1)

        self_attention_code = Code(file_name="fn_self_attention.py", language="Python", font="Monospace", insert_line_no=False,
                            style="dracula", line_spacing=1).scale(0.4).next_to(ex_review_emb, RIGHT)
        self_attention_code.code = remove_invisible_chars(self_attention_code.code)
        self.play(Create(self_attention_code[0]), Write(self_attention_code.code))
        self.wait(1)

        out_token = np.load("ex_review_out.npy")
        ex_review_out = Matrix(np.round(out_token[0], 2), v_buff=1.5, h_buff=1.5).set_row_colors(RED_A, GREEN_A, BLUE_A, ORANGE, YELLOW).scale(0.4).next_to(self_attention_code, RIGHT)
        box = SurroundingRectangle(ex_review_emb, buff=0.1)
        self.play(Create(box))
        self.wait(1)
        self.play(FadeOut(box, target_position=self_attention_code))
        self.play(FadeIn(ex_review_out, target_position=self_attention_code))
        self.wait(1)

        comment = Text("There are two problems here:").scale(0.5).next_to(self_attention_code, DOWN)
        blist = BulletedList("Order of words isn't affecting the output!", "There's no Machine Learning here!!!").scale(0.5).next_to(comment, DOWN)
        self.play(FadeOut(out_eq, W_eq1, W_eq2), Write(comment))
        self.wait(1)
        self.play(Write(blist[0]))
        self.wait(1)
        return super().construct()