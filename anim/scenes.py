import numpy as np
from manim import *
import pickle

from manim_ml.neural_network import NeuralNetwork, FeedForwardLayer

from utils import remove_invisible_chars

class Motivation(Scene):
    def construct(self):
        # self.camera.background_color = "#282a36"
        self.wait(1)
        attention_paper = ImageMobject("anim_data/img/Attention_is_all_you_need.png").to_edge(LEFT)
        self.add(attention_paper)
        self.wait(1)
        gpt1 = SVGMobject("anim_data/img/openai.svg").scale(0.5).next_to(attention_paper).shift(1.5*UP+RIGHT).set_color(WHITE)
        gpt1_text = Text("GPT-1 (Released 2018)").scale(0.5).next_to(gpt1, RIGHT)
        self.add(gpt1)
        self.play(Write(gpt1_text))
        self.wait(1)
        gpt2 = SVGMobject("anim_data/img/openai.svg").scale(0.5).next_to(attention_paper).shift(RIGHT).set_color(WHITE)
        gpt2_text = Text("GPT-2 (Released 2019)").scale(0.5).next_to(gpt2, RIGHT)
        self.add(gpt2)
        self.play(Write(gpt2_text))
        self.wait(1)
        gpt3 = SVGMobject("anim_data/img/openai.svg").scale(0.5).next_to(attention_paper).shift(1.5*DOWN+RIGHT).set_color(WHITE)
        gpt3_text = Text("GPT-3 (Released 2020)").scale(0.5).next_to(gpt3, RIGHT)
        self.add(gpt3)
        self.play(Write(gpt3_text))
        self.wait(1)
        self.remove(attention_paper)
        self.play(VGroup(gpt1, gpt1_text, gpt2, gpt2_text, gpt3, gpt3_text).animate.to_edge(LEFT))
        chatgpt = SVGMobject("anim_data/img/openai.svg").scale(1).set_color(GREEN)
        chatgpt_text = Text("ChatGPT").scale(1).next_to(chatgpt, DOWN)
        self.add(chatgpt)
        self.play(Write(chatgpt_text))
        self.wait(1)
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )
        goal1 = Rectangle(height=5, width=6.5).to_edge(LEFT)
        self.play(Create(goal1))
        self.wait(1)
        goal2 = Rectangle(height=5, width=6.5).to_edge(RIGHT)
        self.play(Create(goal2))
        self.wait(1)
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )
        self.wait(1)

class Dataset(Scene):
    def construct(self):
        # Setting background colour
        # self.camera.background_color = "#282a36"
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
        self.play(FadeOut(out_eq, W_eq1, W_eq2, mat_dim), VGroup(ex_review_emb, word1, word2, word3, word4, word5, self_attention_code, ex_review_out).animate.to_edge(UP).shift(0.5*DOWN))
        self.wait(1)


        emb_token_jum = np.load("ex_review_emb_jum.npy")
        ex_review_emb_jum = Matrix(np.round(emb_token_jum[0], 2), v_buff=1.5, h_buff=1.5).set_row_colors(RED_A, ORANGE, BLUE_A, YELLOW, GREEN_A).scale(0.4).next_to(ex_review_emb, DOWN).shift(0.25*DOWN)
        word1_jum = Text("This", color=RED_A).scale(0.3).next_to(ex_review_emb_jum.get_rows()[0], LEFT).shift(0.3*LEFT)
        word2_jum = Text("very", color=ORANGE).scale(0.3).next_to(ex_review_emb_jum.get_rows()[1], LEFT).shift(0.1*LEFT)
        word3_jum = Text("is", color=BLUE_A).scale(0.3).next_to(ex_review_emb_jum.get_rows()[2], LEFT).shift(0.3*LEFT)
        word4_jum = Text("good", color=YELLOW).scale(0.3).next_to(ex_review_emb_jum.get_rows()[3], LEFT).shift(0.2*LEFT)
        word5_jum = Text("movie", color=GREEN_A).scale(0.3).next_to(ex_review_emb_jum.get_rows()[4], LEFT).shift(0.1*LEFT)
        self.play(Write(word1_jum), Write(word2_jum), Write(word3_jum), Write(word4_jum), Write(word5_jum), Create(ex_review_emb_jum))
        self.wait(1)

        self_attention_code_jum = Code(file_name="fn_self_attention.py", language="Python", font="Monospace", insert_line_no=False,
                            style="dracula", line_spacing=1).scale(0.4).next_to(ex_review_emb_jum, RIGHT)
        self_attention_code_jum.code = remove_invisible_chars(self_attention_code_jum.code)
        self.play(Create(self_attention_code_jum[0]), Write(self_attention_code_jum.code))
        self.wait(1)

        out_token_jum = np.load("ex_review_out_jum.npy")
        ex_review_out_jum = Matrix(np.round(out_token_jum[0], 2), v_buff=1.5, h_buff=1.5).set_row_colors(RED_A, ORANGE, BLUE_A, YELLOW, GREEN_A).scale(0.4).next_to(self_attention_code_jum, RIGHT)
        box_jum = SurroundingRectangle(ex_review_emb_jum, buff=0.1)
        self.play(Create(box_jum))
        self.wait(1)
        self.play(FadeOut(box_jum, target_position=self_attention_code_jum))
        self.play(FadeIn(ex_review_out_jum, target_position=self_attention_code_jum))
        self.wait(1)

        # Positional Encoding
        self.play(FadeOut(self_attention_code, self_attention_code_jum, ex_review_out, ex_review_out_jum))
        pos = np.load("pos.npy")
        add_op = MathTex("+").shift(LEFT)
        pos_mat = Matrix(np.round(pos[0], 2), v_buff=1.5, h_buff=1.5).scale(0.4).next_to(add_op, RIGHT)
        self.play(Write(add_op), Write(pos_mat))
        self.wait(1)
        ex_review_emb_pos = Matrix(np.round(emb_token+pos[0], 2), v_buff=1.5, h_buff=1.5).set_row_colors(RED_A, GREEN_A, BLUE_A, ORANGE, YELLOW).scale(0.4).move_to(ex_review_emb)
        ex_review_emb_jum_pos = Matrix(np.round(emb_token_jum[0]+pos[0], 2), v_buff=1.5, h_buff=1.5).set_row_colors(RED_A, ORANGE, BLUE_A, YELLOW, GREEN_A).scale(0.4).move_to(ex_review_emb_jum)
        self.play(FadeOut(VGroup(add_op, pos_mat), target_position=2*LEFT), ReplacementTransform(VGroup(ex_review_emb, ex_review_emb_jum), VGroup(ex_review_emb_pos, ex_review_emb_jum_pos)))
        self.wait(1)
        pos_mat = Matrix(np.round(pos[0], 2), v_buff=1.5, h_buff=1.5).scale(0.4).to_edge(LEFT).shift(RIGHT)
        self.play(FadeOut(ex_review_emb_pos, word1, word2, word3, word4, word5, ex_review_emb_jum_pos, word1_jum, word2_jum, word3_jum, word4_jum, word5_jum), Create(pos_mat))
        pos_encode_code = Code(file_name="fn_pos_encode.py", language="Python", font="Monospace", insert_line_no=False,
                            style="dracula", line_spacing=1).scale(0.4).next_to(pos_mat, RIGHT)
        pos_encode_code.code = remove_invisible_chars(pos_encode_code.code)
        pos_encode_code.code[2:].set_opacity(0.1)
        self.play(Create(pos_encode_code))
        self.wait(1)
        self.play(pos_encode_code.code[2].animate.set_opacity(1))
        self.wait(1)
        self.play(pos_encode_code.code[3].animate.set_opacity(1))
        self.wait(1)
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )
        self.wait(1)
        
        return super().construct()
    
class QueryKeyValue(Scene):
    def construct(self):
        # Self Attention Operation
        W = MathTex(r"W = \text{softmax}",r"(\frac{X \cdot X^T}{\sqrt{\text{emb}}})").to_edge(UP)
        Y = MathTex(r"Y = W \cdot", r"X").next_to(W, DOWN)
        self.play(Write(W), Write(Y))
        self.wait(1)

        text = Text("There's no Machine Learning here...").scale(0.5).shift(2*DOWN)
        self.play(Write(text))
        self.wait(1)
        self.play(FadeOut(text))

        # Query Key Value
        self.play(W[1][1].animate.scale(1.25).set_color(RED), 
                  W[1][3:5].animate.scale(1.25).set_color(GREEN),
                  Y[1].animate.scale(1.25).set_color(BLUE))
        self.wait(1)
        nn1 = NeuralNetwork([FeedForwardLayer(num_nodes=5),
                            FeedForwardLayer(num_nodes=5)
                            ]).next_to(Y, DOWN)
        nn2 = NeuralNetwork([FeedForwardLayer(num_nodes=5),
                            FeedForwardLayer(num_nodes=5)
                            ]).next_to(nn1, DOWN)
        nn3 = NeuralNetwork([FeedForwardLayer(num_nodes=5),
                            FeedForwardLayer(num_nodes=5)
                            ]).next_to(nn2, DOWN)
        X = MathTex("X").next_to(nn2, LEFT).shift(2*LEFT)
        in1 = Arrow(start = X.get_edge_center(RIGHT), end=nn1.get_edge_center(LEFT))
        in2 = Arrow(start = X.get_edge_center(RIGHT), end=nn2.get_edge_center(LEFT))
        in3 = Arrow(start = X.get_edge_center(RIGHT), end=nn3.get_edge_center(LEFT))
        self.play(Write(X), 
                  W[1][1].animate.scale(1/1.25), 
                  W[1][3:5].animate.scale(1/1.25), 
                  Y[1].animate.scale(1/1.25))
        self.add(nn1, nn2, nn3)
        self.play(Create(in1), Create(in2), Create(in3))
        out1 = Arrow(start = nn1.get_edge_center(RIGHT), end=nn1.get_edge_center(RIGHT)+2*RIGHT)
        out2 = Arrow(start = nn2.get_edge_center(RIGHT), end=nn2.get_edge_center(RIGHT)+2*RIGHT)
        out3 = Arrow(start = nn3.get_edge_center(RIGHT), end=nn3.get_edge_center(RIGHT)+2*RIGHT)
        Q = Text("Query", color=RED).next_to(out1, RIGHT)
        K = Text("Key", color=GREEN).next_to(out2, RIGHT)
        V = Text("Value", color=BLUE).next_to(out3, RIGHT)
        self.play(nn1.make_forward_pass_animation(), run_time=1)
        self.play(Create(out1), Write(Q[0][0]))
        self.play(nn2.make_forward_pass_animation(), run_time=1)
        self.play(Create(out2), Write(K[0][0]))
        self.play(nn3.make_forward_pass_animation(), run_time=1)
        self.play(Create(out3), Write(V[0][0]))
        self.wait(1)
        self.play(Write(Q[1:]), Write(K[1:]), Write(V[1:]))
        self.wait(1)
        W_ = MathTex(r"W = \text{softmax}",r"(\frac{Q \cdot K^T}{\sqrt{\text{emb}}})").to_edge(UP)
        Y_ = MathTex(r"Y = W \cdot", r"V").next_to(W_, DOWN)
        W_[1][1].color = RED
        W_[1][3:5].color = GREEN
        Y_[1].color = BLUE
        self.play(TransformMatchingTex(W, W_), TransformMatchingTex(Y, Y_))
        self.wait(1)
        self.play(FadeOut(X, in1, in2, in3, nn1, nn2, nn3, out1, out2, out3, Q, K, V))
        self.wait(1)

        # Viz Self Attention
        with open ('review1.pkl', 'rb') as fp:
            review1 = pickle.load(fp)
        review1 = Text(review1[:54]).scale(0.5).next_to(Y_, DOWN)
        self.play(Write(review1))
        nn1 = NeuralNetwork([FeedForwardLayer(num_nodes=5),
                            FeedForwardLayer(num_nodes=5)
                            ]).next_to(review1, DOWN)
        nn2 = NeuralNetwork([FeedForwardLayer(num_nodes=5),
                            FeedForwardLayer(num_nodes=5)
                            ]).next_to(nn1, LEFT).shift(LEFT)
        nn3 = NeuralNetwork([FeedForwardLayer(num_nodes=5),
                            FeedForwardLayer(num_nodes=5)
                            ]).next_to(nn1, RIGHT).shift(RIGHT)
        Q = Text("Q", color=RED).scale(0.5).next_to(nn2, DOWN)
        K = Text("K", color=GREEN).scale(0.5).next_to(nn1, DOWN)
        V = Text("V", color=BLUE).scale(0.5).next_to(nn3, DOWN)
        self.play(Create(nn1), Create(nn2), Create(nn3))
        self.play(Write(Q), Write(K), Write(V))
        box = SurroundingRectangle(W_)
        W_op = MathTex(r"\otimes").next_to(Q, RIGHT).shift(0.6*RIGHT)
        connect1 = Line(start=Q.get_edge_center(RIGHT), end=W_op.get_edge_center(LEFT))
        connect2 = Line(start=K.get_edge_center(LEFT), end=W_op.get_edge_center(RIGHT))
        W = MathTex(r"W").scale(0.5).next_to(W_op, DOWN).shift(0.5*DOWN)
        connect3 = Line(start=W_op.get_edge_center(DOWN), end=W.get_edge_center(UP)+0.05*UP)
        self.play(Create(box))
        self.play(ReplacementTransform(box, VGroup(connect1, connect2, W_op, connect3, W)))
        self.wait(1)
        box2 = SurroundingRectangle(Y_)
        Y_op = MathTex(r"\odot").next_to(V, DOWN).shift(0.5*DOWN)
        connect4 = Line(start=W.get_edge_center(RIGHT), end=Y_op.get_edge_center(LEFT))
        connect5 = Line(start=V.get_edge_center(DOWN), end=Y_op.get_edge_center(UP))
        Y = MathTex(r"Y").scale(0.5).next_to(Y_op, DOWN).shift(0.5*DOWN)
        connect6 = Line(start=Y_op.get_edge_center(DOWN), end=Y.get_edge_center(UP)+0.05*UP)
        self.play(Create(box2))
        self.play(ReplacementTransform(box2, VGroup(connect4, connect5, Y_op, connect6, Y)))
        self.wait(1)
        self.play(FadeOut(W_, Y_))
        X_shape = MathTex(r"(\text{num words} \times \text{emb dim})").scale(0.5).next_to(review1, UP)
        K_shape = MathTex(r"(\text{num words} \times \text{emb dim})", color=GREEN).scale(0.5).next_to(K, DOWN).shift(0.55*RIGHT)
        Q_shape = MathTex(r"(\text{num words} \times \text{emb dim})", color=RED).scale(0.5).next_to(Q, LEFT)
        V_shape = MathTex(r"(\text{num words} \times \text{emb dim})", color=BLUE).scale(0.5).next_to(V, RIGHT)
        W_shape = MathTex(r"(\text{num words} \times \text{num words})").scale(0.5).next_to(W, DOWN).shift(RIGHT)
        Y_shape = MathTex(r"(\text{num words} \times \text{emb dim})").scale(0.5).next_to(Y, RIGHT)
        self.play(Write(X_shape))
        self.wait(1)
        self.play(Write(K_shape), Write(Q_shape), Write(V_shape))
        self.wait(1)
        self.play(Write(W_shape))
        self.wait(1)
        self.play(VGroup(review1, Q, K, V, W_op, 
                         connect1, connect2, connect3, connect4, connect5, connect6,
                         W, Y, Y_op, X_shape, K_shape, Q_shape, V_shape, W_shape, Y_shape).animate.to_edge(UP),
                         nn1.animate.shift(1.6*UP), nn2.animate.shift(1.6*UP), nn3.animate.shift(1.6*UP))
        W_viz = ImageMobject("img/W_viz.png").scale(0.8).next_to(W, LEFT).to_edge(DOWN).shift(0.35*LEFT+0.25*DOWN)
        self.play(FadeIn(W_viz))
        self.wait(1)
        circ1 = Circle(radius=0.3, color=GREEN).shift(2.64*DOWN+5.4*LEFT)
        self.play(Create(circ1))
        self.wait(1)
        circ2 = Circle(radius=0.3, color=GREEN).shift(3.4*DOWN+3.25*LEFT)
        self.play(Create(circ2))
        self.wait(1)
        rect2 = Rectangle(height=0.4, width=4.5, color=RED).shift(4.5*LEFT+0.45*DOWN)
        self.play(Create(rect2))
        self.wait(1)
        rect1 = Rectangle(height=0.4, width=4.5, color=RED).shift(4.5*LEFT+1.5*DOWN)
        self.play(Create(rect1))
        self.wait(1)
        self.play(Write(Y_shape))
        self.wait(1)
        self.play(FadeOut(VGroup(review1, Q, K, V, W_op, 
                         connect1, connect2, connect3, connect4, connect5, connect6,
                         W, Y, Y_op, X_shape, K_shape, Q_shape, V_shape, W_shape, Y_shape), nn1, nn2, nn3))
        comment = Text("What about this lack of attention to detail???").scale(0.5).next_to(W_viz, RIGHT)
        self.play(Write(comment))
        self.wait(1)
        title = Text("Multi-Head Attention").to_edge(UP)
        self.play(Write(title), FadeOut(W_viz, comment, circ1, circ2, rect1, rect2))
        self.wait(1)
        return super().construct()
    
class MultiHeadAttention(Scene):
    def construct(self):
        # Title
        title = Text("Multi-Head Attention").to_edge(UP)
        self.add(title)
        self.wait(1)
        self.play(title.animate.scale(0.5).shift(3.5*RIGHT))
        # Single Head Attention Intro
        title2 = Text("Single-Head Attention").scale(0.5).next_to(title, LEFT).shift(3.25*LEFT)
        self.play(Write(title2))
        self.wait(1)
        X_dim = MathTex(r"(t \times k)").scale(0.75).next_to(title2, DOWN).shift(0.25*DOWN)
        X = MathTex("X").next_to(X_dim, DOWN)
        self.play(Write(X_dim), Write(X))
        nn1 = RoundedRectangle(corner_radius=0.15, height=0.5, width=1, color=GREEN).next_to(X, DOWN).shift(0.25*DOWN)
        nn2 = RoundedRectangle(corner_radius=0.15, height=0.5, width=1, color=RED).next_to(nn1, RIGHT)
        nn3 = RoundedRectangle(corner_radius=0.15, height=0.5, width=1, color=BLUE).next_to(nn1, LEFT)
        box = SurroundingRectangle(VGroup(nn1, nn2, nn3), color=WHITE)
        self.play(Create(box), Create(nn1), Create(nn2), Create(nn3))
        self.wait(1)
        Q = Text("Q", color=RED).scale(0.5).next_to(nn2, DOWN)
        K = Text("K", color=GREEN).scale(0.5).next_to(nn1, DOWN)
        V = Text("V", color=BLUE).scale(0.5).next_to(nn3, DOWN)
        K_shape = MathTex(r"(t \times k)", color=GREEN).scale(0.75).next_to(K, DOWN)
        Q_shape = MathTex(r"(t \times k)", color=RED).scale(0.75).next_to(Q, DOWN)
        V_shape = MathTex(r"(t \times k)", color=BLUE).scale(0.75).next_to(V, DOWN)
        self.play(Write(VGroup(Q, K, V, Q_shape, K_shape, V_shape)))
        self.wait(1)
        self_attention = Text("Self Attention").scale(0.5).next_to(K_shape, DOWN)
        sa_box = SurroundingRectangle(self_attention, color=WHITE)
        Y = MathTex(r"Y").scale(0.75).next_to(sa_box, DOWN)
        Y_shape = MathTex(r"(t \times k)").scale(0.75).next_to(Y, DOWN)
        self.play(Create(VGroup(self_attention, sa_box, Y, Y_shape)))
        self.wait(1)
        # Multi Head Attention
        X_dim_ = MathTex(r"(t \times k)").scale(0.75).next_to(title, DOWN).shift(0.25*DOWN)
        X_ = MathTex("X").next_to(X_dim_, DOWN)
        self.play(Write(X_dim_), Write(X_))
        nn1_ = RoundedRectangle(corner_radius=0.15, height=0.5, width=1, color=GREEN).next_to(X_, DOWN).shift(0.25*DOWN)
        nn2_ = RoundedRectangle(corner_radius=0.15, height=0.5, width=1, color=RED).next_to(nn1_, RIGHT)
        nn3_ = RoundedRectangle(corner_radius=0.15, height=0.5, width=1, color=BLUE).next_to(nn1_, LEFT)
        box_ = SurroundingRectangle(VGroup(nn1_, nn2_, nn3_), color=WHITE)
        self.play(Create(box_), Create(nn1_), Create(nn2_), Create(nn3_))
        self.wait(1)
        Q_ = Text("Q", color=RED).scale(0.5).next_to(nn2_, DOWN)
        K_ = Text("K", color=GREEN).scale(0.5).next_to(nn1_, DOWN)
        V_ = Text("V", color=BLUE).scale(0.5).next_to(nn3_, DOWN)
        K_shape_ = MathTex(r"(t \times k)", color=GREEN).scale(0.75).next_to(K_, DOWN)
        Q_shape_ = MathTex(r"(t \times k)", color=RED).scale(0.75).next_to(Q_, DOWN)
        V_shape_ = MathTex(r"(t \times k)", color=BLUE).scale(0.75).next_to(V_, DOWN)
        self.play(Write(VGroup(Q_, K_, V_, Q_shape_, K_shape_, V_shape_)))
        self.wait(1)
        K_shape__ = MathTex(r"(t \times \frac{k}{r} \times r)", color=GREEN).scale(0.45).next_to(K_, DOWN)
        Q_shape__ = MathTex(r"(t \times \frac{k}{r} \times r)", color=RED).scale(0.45).next_to(Q_, DOWN)
        V_shape__ = MathTex(r"(t \times \frac{k}{r} \times r)", color=BLUE).scale(0.45).next_to(V_, DOWN)
        self.play(TransformMatchingTex(VGroup(K_shape_, Q_shape_, V_shape_), VGroup(K_shape__, Q_shape__, V_shape__)))
        self.wait(1)
        Q1_ = Text("Q1", color=RED).scale(0.35).next_to(nn2_, DOWN)
        K1_ = Text("K1", color=GREEN).scale(0.35).next_to(nn1_, DOWN)
        V1_ = Text("V1", color=BLUE).scale(0.35).next_to(nn3_, DOWN)
        Qdot_ = MathTex(r"\vdots", color=RED).scale(0.35).next_to(Q1_, DOWN)
        Kdot_ = MathTex(r"\vdots", color=GREEN).scale(0.35).next_to(K1_, DOWN)
        Vdot_ = MathTex(r"\vdots", color=BLUE).scale(0.35).next_to(V1_, DOWN)
        Qr_ = Text("Qr", color=RED).scale(0.35).next_to(Qdot_, DOWN)
        Kr_ = Text("Kr", color=GREEN).scale(0.35).next_to(Kdot_, DOWN)
        Vr_ = Text("Vr", color=BLUE).scale(0.35).next_to(Vdot_, DOWN)
        K_shape___ = MathTex(r"(t \times \frac{k}{r})", color=GREEN).scale(0.35).next_to(Kr_, DOWN)
        Q_shape___ = MathTex(r"(t \times \frac{k}{r})", color=RED).scale(0.35).next_to(Qr_, DOWN)
        V_shape___ = MathTex(r"(t \times \frac{k}{r})", color=BLUE).scale(0.35).next_to(Vr_, DOWN)
        self.play(ReplacementTransform(VGroup(K_shape__, Q_shape__, V_shape__, Q_, K_, V_), VGroup(Q1_, K1_, V1_, Qdot_, Kdot_, Vdot_, Qr_, Kr_, Vr_, K_shape___, Q_shape___, V_shape___)))
        self.wait(1)
        self_attention_ = Text("r Parallel Self Attentions").scale(0.5).next_to(K_shape___, DOWN)
        sa_box_ = SurroundingRectangle(self_attention_, color=WHITE)
        Y1_ = MathTex(r"Y1").scale(0.35).next_to(V_shape___, DOWN).shift(0.5*DOWN)
        Y1_shape = MathTex(r"(t \times \frac{k}{r})").scale(0.5).next_to(Y1_, DOWN)
        Y2_ = MathTex(r"Y2").scale(0.35).next_to(Y1_, RIGHT)
        Y3_ = MathTex(r"Y3").scale(0.35).next_to(Y2_, RIGHT)
        cdots = MathTex(r"\cdots").next_to(Y3_, RIGHT)
        Yr_ = MathTex(r"Yr").scale(0.35).next_to(cdots, RIGHT)
        Yr_shape = MathTex(r"(t \times \frac{k}{r})").scale(0.5).next_to(Yr_, DOWN)
        self.play(Create(VGroup(self_attention_, sa_box_, Y1_, Y1_shape, Y2_, Y3_, cdots, Yr_, Yr_shape)))
        self.wait(1)
        Y_ = MathTex(r"Y").scale(0.75).next_to(sa_box_, DOWN)
        Y_shape_ = MathTex(r"(t \times k)").scale(0.75).next_to(Y_, RIGHT).shift(RIGHT)
        self.play(ReplacementTransform(VGroup(Y1_, Y1_shape, Y2_, Y3_, cdots, Yr_, Yr_shape), VGroup(Y_, Y_shape_)))
        self.wait(1)
        unify = RoundedRectangle(corner_radius=0.15, height=0.5, width=3, color=YELLOW).next_to(Y_, DOWN)
        Y__ = MathTex(r"Y").scale(0.75).next_to(unify, DOWN)
        Y_shape__ = MathTex(r"(t \times k)").scale(0.75).next_to(Y__, RIGHT).shift(RIGHT)
        self.play(Create(unify), Write(Y__), Write(Y_shape__))
        self.wait(1)
        self.play(FadeOut(title2, X_dim, X, box, nn1, nn2, nn3, VGroup(Q, K, V, Q_shape, K_shape, V_shape), VGroup(self_attention, sa_box, Y, Y_shape, sa_box)))
        self.wait(1)
        comment = Text("r different attention weight matrices!").scale(0.65).next_to(Vdot_, LEFT).shift(0.5*LEFT)
        self.play(Write(comment))
        self.wait(1)
        W_viz1 = ImageMobject("img/W_viz_4heads_1.png").scale(0.6).to_edge(LEFT).shift(1.5*UP)
        W_viz2 = ImageMobject("img/W_viz_4heads_2.png").scale(0.6).next_to(W_viz1, RIGHT)
        W_viz3 = ImageMobject("img/W_viz_4heads_3.png").scale(0.6).next_to(W_viz1, DOWN)
        W_viz4 = ImageMobject("img/W_viz_4heads_4.png").scale(0.6).next_to(W_viz3, RIGHT)
        self.play(FadeOut(comment), FadeIn(W_viz1, W_viz2, W_viz3, W_viz4))
        self.wait(1)
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )
        self.wait(1)
        return super().construct()
    
class CodingMultiHeadAttention(Scene):
    def construct(self):
        # Loading code
        self_attention_code = Code(file_name="fn_multi_attention.py", language="Python", font="Monospace", insert_line_no=False,
                            style="dracula", line_spacing=1).scale(0.3).to_edge(RIGHT)
        self_attention_code.code = remove_invisible_chars(self_attention_code.code)
        self.play(Create(self_attention_code[0]), Create(self_attention_code.code[:6]), Create(self_attention_code.code[17]))
        
        # Multi Head Attention
        X_dim_ = MathTex(r"(b \times t \times k)").scale(0.75).to_edge(UP).to_edge(LEFT).shift(0.25*DOWN+2.25*RIGHT)
        X_ = MathTex("X").next_to(X_dim_, DOWN)
        self.play(Write(X_dim_), Write(X_), Write(self_attention_code.code[18]))
        self.wait(1)
        self.play(Write(self_attention_code.code[6:9]))
        self.wait(1)
        nn1_ = RoundedRectangle(corner_radius=0.15, height=0.5, width=2, color=GREEN).next_to(X_, DOWN).shift(0.25*DOWN)
        nn2_ = RoundedRectangle(corner_radius=0.15, height=0.5, width=2, color=RED).next_to(nn1_, RIGHT)
        nn3_ = RoundedRectangle(corner_radius=0.15, height=0.5, width=2, color=BLUE).next_to(nn1_, LEFT)
        box_ = SurroundingRectangle(VGroup(nn1_, nn2_, nn3_), color=WHITE)
        self.play(Create(box_), Create(nn1_), Create(nn2_), Create(nn3_), Write(self_attention_code.code[9:14]))
        self.wait(1)
        Q_ = Text("Q", color=RED).scale(0.5).next_to(nn2_, DOWN)
        K_ = Text("K", color=GREEN).scale(0.5).next_to(nn1_, DOWN)
        V_ = Text("V", color=BLUE).scale(0.5).next_to(nn3_, DOWN)
        K_shape_ = MathTex(r"(b \times t \times k)", color=GREEN).scale(0.75).next_to(K_, DOWN)
        Q_shape_ = MathTex(r"(b \times t \times k)", color=RED).scale(0.75).next_to(Q_, DOWN)
        V_shape_ = MathTex(r"(b \times t \times k)", color=BLUE).scale(0.75).next_to(V_, DOWN)
        self.play(Write(VGroup(Q_, K_, V_, Q_shape_, K_shape_, V_shape_)), Write(self_attention_code.code[20:26]))
        self.wait(1)
        K_shape__ = MathTex(r"(b \times t \times h \times \frac{k}{h})", color=GREEN).scale(0.65).next_to(K_, DOWN)
        Q_shape__ = MathTex(r"(b \times t \times h \times \frac{k}{h})", color=RED).scale(0.65).next_to(Q_, DOWN)
        V_shape__ = MathTex(r"(b \times t \times h \times \frac{k}{h})", color=BLUE).scale(0.65).next_to(V_, DOWN)
        self.play(TransformMatchingTex(VGroup(K_shape_, Q_shape_, V_shape_), VGroup(K_shape__, Q_shape__, V_shape__)), Write(self_attention_code.code[26:31]))
        self.wait(1)
        Q1_ = Text("Q1", color=RED).scale(0.35).next_to(nn2_, DOWN)
        K1_ = Text("K1", color=GREEN).scale(0.35).next_to(nn1_, DOWN)
        V1_ = Text("V1", color=BLUE).scale(0.35).next_to(nn3_, DOWN)
        Qdot_ = MathTex(r"\vdots", color=RED).scale(0.35).next_to(Q1_, DOWN)
        Kdot_ = MathTex(r"\vdots", color=GREEN).scale(0.35).next_to(K1_, DOWN)
        Vdot_ = MathTex(r"\vdots", color=BLUE).scale(0.35).next_to(V1_, DOWN)
        Qr_ = Text("Qr", color=RED).scale(0.35).next_to(Qdot_, DOWN)
        Kr_ = Text("Kr", color=GREEN).scale(0.35).next_to(Kdot_, DOWN)
        Vr_ = Text("Vr", color=BLUE).scale(0.35).next_to(Vdot_, DOWN)
        K_shape___ = MathTex(r"(bh \times t \times \frac{k}{h})", color=GREEN).scale(0.65).next_to(Kr_, DOWN)
        Q_shape___ = MathTex(r"(bh \times t \times \frac{k}{h})", color=RED).scale(0.65).next_to(Qr_, DOWN)
        V_shape___ = MathTex(r"(bh \times t \times \frac{k}{h})", color=BLUE).scale(0.65).next_to(Vr_, DOWN)
        self.play(ReplacementTransform(VGroup(K_shape__, Q_shape__, V_shape__, Q_, K_, V_), 
                                       VGroup(Q1_, K1_, V1_, Qdot_, Kdot_, Vdot_, Qr_, Kr_, Vr_, K_shape___, Q_shape___, V_shape___)),
                                       Write(self_attention_code.code[31:36]))
        self.wait(1)
        self_attention_ = Text("Parallel Self Attentions").scale(0.75).next_to(K_shape___, DOWN)
        sa_box_ = SurroundingRectangle(self_attention_, color=WHITE)
        Y1_ = MathTex(r"Y1").scale(0.75).next_to(V_shape___, DOWN).shift(1*DOWN)
        Y1_shape = MathTex(r"(b \times h \times t \times \frac{k}{h})").scale(0.5).next_to(Y1_, DOWN)
        Y2_ = MathTex(r"Y2").scale(0.75).next_to(Y1_, RIGHT).shift(0.4*RIGHT)
        Y3_ = MathTex(r"Y3").scale(0.75).next_to(Y2_, RIGHT).shift(0.4*RIGHT)
        cdots = MathTex(r"\cdots").next_to(Y3_, RIGHT)
        Yr_ = MathTex(r"Yr").scale(0.75).next_to(cdots, RIGHT).shift(0.4*RIGHT)
        Yr_shape = MathTex(r"(b \times h \times t \times \frac{k}{h})").scale(0.5).next_to(Yr_, DOWN)
        self.play(Create(VGroup(self_attention_, sa_box_, Y1_, Y1_shape, Y2_, Y3_, cdots, Yr_, Yr_shape)), Write(self_attention_code.code[36:41]))
        self.wait(1)
        Y_ = MathTex(r"Y").scale(0.75).next_to(sa_box_, DOWN)
        Y_shape_ = MathTex(r"(b \times t \times k)").scale(0.75).next_to(Y_, RIGHT).shift(RIGHT)
        self.play(ReplacementTransform(VGroup(Y1_, Y1_shape, Y2_, Y3_, cdots, Yr_, Yr_shape), VGroup(Y_, Y_shape_)), Write(self_attention_code.code[41:43]))
        self.wait(1)
        unify = RoundedRectangle(corner_radius=0.15, height=0.5, width=3, color=YELLOW).next_to(Y_, DOWN)
        self.play(Create(unify), Write(self_attention_code.code[14:16]))
        Y__ = MathTex(r"Y").scale(0.75).next_to(unify, DOWN)
        Y_shape__ = MathTex(r"(b \times t \times k)").scale(0.75).next_to(Y__, RIGHT).shift(RIGHT)
        self.play(Write(self_attention_code.code[42:]), Write(Y__), Write(Y_shape__))
        self.wait(1)
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )
        self.wait(1)
        return super().construct()
    
class Transformer(Scene):
     def construct(self):
        # Loading code
        transformer_code = Code(file_name="fn_transformer.py", language="Python", font="Monospace", insert_line_no=False,
                            style="dracula", line_spacing=1).scale(0.3).to_edge(RIGHT)
        transformer_code.code = remove_invisible_chars(transformer_code.code)
        self.play(Create(transformer_code[0]), Create(transformer_code.code[:4]), Create(transformer_code.code[25]))
        self.wait(1)
        
        # 5 Steps to Transformers
        title = Text("5 Steps to Transformers").scale(0.85).to_edge(UP).to_edge(LEFT)
        self.play(Write(title))
        bp = BulletedList("Token Embedding", 
                          "Positional Embedding",
                          "Self Attention",
                          "Fully Connected Network",
                          "Output").scale(0.75).next_to(title,DOWN).shift(0.5*DOWN)
        self.play(Write(bp[0]), Write(transformer_code.code[4:7]), Write(transformer_code.code[26:30]))
        self.wait(1)
        self.play(Write(bp[1]), Write(transformer_code.code[7:10]), Write(transformer_code.code[30:34]))
        self.wait(1)
        self.play(Write(bp[2]), Write(transformer_code.code[10:12]), Write(transformer_code.code[34:36]))
        self.wait(1)
        self.play(Write(bp[3]), Write(transformer_code.code[12:23]), Write(transformer_code.code[36:40]))
        self.wait(1)
        self.play(Write(bp[4]), Write(transformer_code.code[23]), Write(transformer_code.code[40:]))
        self.wait(1)
        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )
        self.wait(1)

        # Training
        # Loading code
        training_code = Code(file_name="fn_train.py", language="Python", font="Monospace", insert_line_no=False,
                            style="dracula", line_spacing=1).scale(0.3).to_edge(RIGHT)
        training_code.code = remove_invisible_chars(training_code.code)
        self.play(Create(training_code[0]))
        self.play(Write(training_code.code))
        self.wait(1)
        acc = np.load("acc.npy")
        
        def get_acc(x):
            return acc[x]
            
        ax = Axes(
            x_range=[0, len(acc)-1, 1], y_range=[0.4, 1, 0.1], # [start, stop, step_size]
            y_axis_config={"include_tip": False, "include_numbers": True},
            x_axis_config={"include_numbers": False, "include_ticks": False, 
                           "include_tip": True, "label_direction": DOWN}, 
            tips=False
            )
        labels = ax.get_axis_labels(x_label="Epochs", y_label="Accuracy")
        VGroup(ax, labels).scale(0.5).to_edge(LEFT)
        graph = ax.plot(lambda x: get_acc(int(x)), stroke_width=1)
        self.play(Create(VGroup(ax, labels)))
        self.wait(1)
        self.play(Write(graph), run_time=20)
        self.wait(1)
        return super().construct()