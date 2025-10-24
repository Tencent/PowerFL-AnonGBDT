
#ifndef SPU_ANON_XGB_H
#define SPU_ANON_XGB_H

#include <utility>
#include "libspu/mpc/cheetah/cpsi/vole_cpsi.h"
#include "psi/utils/serialize.h"
#include "libspu/mpc/cheetah/cpsi/operators.h"
#include "libspu/mpc/cheetah/cpsi/histogram.h"
#include <memory>

namespace spu::mpc::cheetah {
    class GradHess {
    public:
        GradHess() = default;
        GradHess(uint64_t gg, uint64_t hh);

        GradHess(const std::vector<uint64_t> &ggs, const std::vector<uint64_t> &hhs);

        GradHess(const GradHess &other);

        GradHess extend(size_t len);

        size_t size() const {
            return grad_.size();
        }

        GradHess clone() const {
            return GradHess(this->getGrad(), this->getHess());
        }

        GradHess sum() const {
            uint64_t grad = std::accumulate(grad_.begin(), grad_.end(), 0ULL);
            uint64_t hess = std::accumulate(hess_.begin(), hess_.end(), 0ULL);
            return GradHess(grad, hess);
        }

        const GradHess &operator=(const GradHess &other);

        const GradHess &operator+(const GradHess &other);

        const GradHess &operator-(const GradHess &other);

        static GradHess concat(const GradHess &aa, const GradHess &bb);

        static GradHess concat(const std::vector<GradHess> &arr);

        GradHess slice(size_t bgn, size_t end) const {
            SPU_ENFORCE(bgn <= this->size() && end <= this->size(), "begin = {}, end = {}, size = {}", bgn, end, this->size());

            std::vector<uint64_t> grad(grad_.begin() + bgn, grad_.begin() + end);
            std::vector<uint64_t> hess(hess_.begin() + bgn, hess_.begin() + end);

            return GradHess(grad, hess);
        }

        std::vector<uint64_t> getGrad() const {
            return grad_;
        }

        std::vector<uint64_t> getHess() const {
            return hess_;
        }

    private:
        std::vector<uint64_t> grad_;
        std::vector<uint64_t> hess_;
    };



    class AnonXGB {
    public:
    
        AnonXGB(std::shared_ptr<yacl::link::Context> link_ctx, psi::PsiRoleType role, 
                size_t log2_scale, bool is_inference=false, bool enable_sigmoid=true);

        GradHess multiplex(const GradHess &grad_hess_ashr, const std::vector<uint8_t> &masks, int type = 0);

        GradHess self_share(const GradHess &encoded_grad_hess, int pos);

        GradHess opprf_share(const GradHess &encoded_grad_hess, int pos);

        void init_histogram();

        void init_histogram(std::shared_ptr<HistoGram> &histogram_ptr0, std::shared_ptr<HistoGram> &histogram_ptr1);

        void initialize(const std::vector<std::string> &items, const std::vector<double> &labels, size_t cols,
                        bool init_ckks=true);

        void initialize_inference(const std::vector<std::string> &items, size_t cols);

        void adjust_scale(GradHess &grad_hess_ashr, double new_scale);

        std::vector<std::vector<uint8_t>> adjust_split_bools(const std::vector<std::vector<uint8_t>> &split_bools,
                                                             std::shared_ptr<VoleCPsi> vole_cpsi_ptr);

        double compute_histogram(GradHess &grad_hess_ashr, const std::vector<std::vector<uint8_t>> &split_bools, int pos);

        double compute_histogram(std::vector<GradHess> &grad_hess_ashr_vec, 
                                 const std::vector<std::vector<uint8_t>> &split_bools,
                                 int pos);

        size_t split_node(const std::vector<std::vector<uint8_t>> &split_bools, double lambda,
                          const GradHess &grad_hess_ashr0, const GradHess &grad_hess_ashr1);

        void compute_histogram(std::vector<GradHess> &hist_gh_ashr_vec0,
                               std::vector<GradHess> &hist_gh_ashr_vec1,
                               const std::vector<std::vector<uint8_t>> &split_bools);

        std::vector<size_t> find_optimal_cols(double lambda,
                                              const std::vector<GradHess> &lhist_ashr_vec0,
                                              const std::vector<GradHess> &lhist_ashr_vec1,
                                              const std::vector<GradHess> &gh_ashr_vec0,
                                              const std::vector<GradHess> &gh_ashr_vec1);

        void share_bool_splits(std::vector<std::vector<uint8_t>> &bool0_vec, std::vector<std::vector<uint8_t>> &bool1_vec,
                               const std::vector<std::vector<uint8_t>> &split_bools,
                               const std::vector<size_t> &index_arr);



        void share_bool_splits_and_multiplex(std::vector<GradHess> &gh_ashr0, std::vector<GradHess> &gh_ashr1, 
                                             const std::vector<std::vector<uint8_t>> &split_bools,
                                             const std::vector<size_t> &index_arr);

        void multiplex(std::vector<GradHess> &grad_hess_ashr_vec, 
                       const std::vector<std::vector<uint8_t>> &mask_vec,
                       int type = 0);

        std::vector<uint64_t> multiplex_single(const std::vector<uint64_t> &shares,
                                               const std::vector<std::vector<uint8_t>> &mask_vec);

        void update_tree_eval(const std::vector<std::vector<uint8_t>> &leaf_mask_bshr_vec0,
                              const std::vector<std::vector<uint8_t>> &leaf_mask_bshr_vec1,
                              const std::vector<uint64_t> &leaf_wt_ashr);

        void gen_tree(const std::vector<std::vector<uint8_t>> &split_bools, double lambda,
                      int tree_level, bool is_last_tree = false);

        void infer_acc_weights_with_lookup_table(
          size_t num_leaves, 
          size_t num_bins,
          vector<uint64_t> &packed_indicators_sender,
          vector<vector<uint8_t>> &indicators_receiver,
          std::vector<uint64_t> &weight,
          vector<uint64_t> &scores
        );

        void infer_acc_weights_with_mux(
          size_t num_leaves, 
          size_t num_bins,
          vector<uint64_t> &packed_indicators_sender,
          vector<vector<uint8_t>> &indicators_receiver,
          std::vector<uint64_t> &weight,
          vector<uint64_t> &scores
        );
        
        void infer_acc_weights(
          size_t num_leaves, 
          size_t num_bins,
          vector<uint64_t> &packed_indicators_sender,
          vector<vector<uint8_t>> &indicators_receiver,
          std::vector<uint64_t> &weight,
          vector<uint64_t> &scores,
          bool enable_lookup_table_cal_weight
        ) ;

        std::vector<uint64_t> infer(const size_t num_tree, 
                                const size_t depth,    
                                std::vector<std::pair<std::vector<std::vector<std::tuple<int, double, double>>>,std::vector<uint64_t>>> &model,
                                std::vector<std::vector<double>> &feats,
                                bool enable_lookup_table_cal_weight=true);

        std::pair<std::vector<uint64_t>, std::vector<uint64_t>> genLookUpTableForWeight(
            double weight, 
            std::vector<uint8_t> &plaintext_indicator, 
            std::vector<uint8_t> &shared_indicator);

        std::vector<double> recoveryScoreInInference(const size_t &n, std::vector<uint64_t> &scores);

        std::vector<uint64_t> compute_gain(const std::vector<uint64_t> &g_shr,
                                                const std::vector<uint64_t> &h_shr);

        std::vector<uint8_t> msb(const std::vector<uint64_t> &shares, size_t nbits=64);
        
        std::vector<uint8_t> msbsigmoid(const std::vector<uint64_t> &shares, uint64_t low_seg, 
                                        uint64_t high_seg, size_t skip_bits, size_t nbits = 64);

        void multiplex(std::vector<uint64_t> &gshares, std::vector<uint64_t> &hshares, 
                       const std::vector<uint8_t> &masks, int type = 0);

        std::vector<uint64_t> multiplex(const std::vector<uint8_t> &masks, const std::vector<uint64_t> &shares0);

        std::vector<uint64_t> mux(const std::vector<uint8_t> &masks, const std::vector<uint64_t> &shares0,
                                  const std::vector<uint64_t> &shares1);

        std::vector<uint64_t> add(const std::vector<uint64_t> &shares0, const std::vector<uint64_t> &shares1);

        std::vector<uint64_t> sub(const std::vector<uint64_t> &shares0, const std::vector<uint64_t> &shares1);

        std::vector<uint64_t> mul(const std::vector<uint64_t> &shares0, const std::vector<uint64_t> &shares1);

        void mul(std::vector<uint64_t> &inputs);

        std::vector<uint64_t> mul(const std::vector<double> &inputs);

        std::vector<uint64_t> square(const std::vector<uint64_t> &shares);

        void truncate(std::vector<uint64_t> &ashr, size_t log2_scale, SignType sign);

        void negate(std::vector<uint64_t> &shares);

        void add_plain(std::vector<uint64_t> &shares, const std::vector<double> &plains, size_t log2_scale);

        void mul_plain(std::vector<uint64_t> &shares, double factor,
                       size_t log2_scale, SignType sign);

        std::vector<uint8_t> bits_xor(const std::vector<uint8_t> &shares0, const std::vector<uint8_t> &shares1);

        void add_plain(std::vector<uint64_t> &shares, double plain, double scale);

        std::vector<uint8_t> is_left(const std::vector<uint64_t> &idx_ashr);

        std::vector<uint64_t> recoveryA(const std::vector<uint64_t> &ashr);

        template<typename T>
        std::vector<T> recoveryB(const std::vector<T> &bshr);

        std::vector<uint64_t> sigmoid(const std::vector<uint64_t> &shares, uint8_t type=2);

        std::vector<uint64_t> divide(const std::vector<uint64_t> &a_shr,
                                     const std::vector<uint64_t> &babs_shr, SignType sign);

        std::vector<uint64_t> square_and_divide(const std::vector<uint64_t> &a_shr,
                                                const std::vector<uint64_t> &babs_shr);

        std::vector<uint64_t> arg_max(const std::vector<uint64_t> &a_shr, size_t cols = 0);

        std::vector<uint64_t> bitwise_and(const std::vector<uint64_t> &values, size_t bitwidth);

        std::shared_ptr<BasicOTProtocols> get_base_ot() const {
            return base_ot_;
        }

        std::vector<uint8_t> get_mask0() const {
            return mask0_;
        }

        std::vector<uint8_t> get_mask1() const {
            return mask1_;
        }

        GradHess get_grad_hess_ashr0() const {
            return grad_hess_ashr0_;
        }

        GradHess get_grad_hess_ashr1() const {
            return grad_hess_ashr1_;
        }

        size_t get_log2_scale() const {
            return log2_scale_;
        }

        psi::PsiRoleType get_role() const {
          return role_;
        }

        std::shared_ptr<yacl::link::Context> get_link_context() const {
            return link_ctx_;
        }

        size_t get_cols() const {
            return self_cols_ + peer_cols_;
        }

        size_t get_peer_cols() const {
            return peer_cols_;
        }

        std::array<std::shared_ptr<HistoGram>, 2> get_histograms() const {
            return {histogram_ptr0_, histogram_ptr1_};
        }

        void update_grad_hess_ashr();

        uint64_t encode(double vv, size_t log2_scale) const {
            auto ww = static_cast<int64_t>(std::round(vv * std::pow(2.0, log2_scale)));
            return static_cast<uint64_t>(ww);
        }

        double decode(uint64_t vv, size_t log2_scale) const {
            return static_cast<int64_t>(vv) / std::pow(2.0, log2_scale);
        }




        // single cpsi
        AnonXGB(std::shared_ptr<yacl::link::Context> link_ctx, psi::PsiRoleType role, size_t log2_scale, int single);

        void initializeSingle(const std::vector<std::string> &items, const std::vector<double> &labels, size_t cols);

        void compute_histogramSingle(std::vector<GradHess> &grad_hess_ashr_vec, 
                                     const std::vector<std::vector<uint8_t>> &split_bools);

        void compute_histogramSingle(std::vector<GradHess> &hist_gh_ashr_vec0,
                                     std::vector<GradHess> &hist_gh_ashr_vec1,
                                     const std::vector<std::vector<uint8_t>> &split_bools);

        std::vector<uint8_t> synchron_col(size_t col_index);

        void share_bool_splitsSingle(std::vector<std::vector<uint8_t>> &bool_vec,
                                     const std::vector<std::vector<uint8_t>> &split_bools,
                                     const std::vector<size_t> &index_arr);

        void update_tree_evalSingle(const std::vector<std::vector<uint8_t>> &leaf_mask_bshr_vec,
                                    const std::vector<uint64_t> &leaf_wt_ashr);

        void update_grad_hess_ashrSingle();

        std::pair<std::vector<std::vector<size_t>>, std::vector<uint64_t>> 
        gen_treeSingle(const std::vector<std::vector<uint8_t>> &split_bools, double lambda,
                       int tree_level, bool is_last_tree = false);

    
        std::atomic<size_t> sync_indicator_comm_bytes = 0;
        std::atomic<size_t> update_indicator_comm_bytes = 0;

    private:
        void fill_tables_ot_sender_for_infer(
            size_t num_bins,
            vector<vector<uint8_t>> &indicators_receiver, 
            vector<vector<uint8_t>> &indicators_sender,
            vector<uint64_t> &tables, 
            vector<vector<uint64_t>> &rands, 
            std::vector<uint64_t> &weight,
            vector<uint8_t> &sel,
            size_t node_ind
        );

        void fill_tables_ot_receiver_for_infer(
            size_t num_bins,
            vector<vector<uint8_t>> &indicators_sender,
            vector<uint64_t> tables,
            vector<vector<uint64_t>> &rands, 
            std::vector<uint64_t> &weights,
            size_t node_ind
        );

        size_t ExchangeSize(size_t size) const {
            link_ctx_->SendAsyncThrottled(
                    link_ctx_->NextRank(), psi::utils::SerializeSize(size),
                    fmt::format("AnonXGB:SELF_SIZE={}", size));

            size_t peer_size = psi::utils::DeserializeSize(
                    link_ctx_->Recv(link_ctx_->NextRank(), fmt::format("AnonXGB:PEER_SIZE")));

            return peer_size;
        }

        void grad_hess_sum_test(GradHess &grad_hess_ashr0, GradHess &grad_hess_ashr1, const std::string &str);

        std::vector<uint64_t> encode(const std::vector<double> &arr, size_t log2_scale) const {
            double scale = std::pow(2.0, log2_scale);
            std::vector<uint64_t> res(arr.size());
            std::transform(arr.begin(), arr.end(), res.begin(), [&](double vv) {
                auto ww = static_cast<int64_t>(std::round(vv * scale));
                return static_cast<uint64_t>(ww);
            });
            return res;
        }

        std::vector<uint64_t> sigmoid1(const std::vector<uint64_t> &shares);
        std::vector<uint64_t> sigmoid2(const std::vector<uint64_t> &shares);

        void init_bool_vectors(std::vector<uint64_t>& bool0, std::vector<uint64_t>& bool1,
                               std::vector<size_t>& idx_pos_arr0, std::vector<size_t>& idx_pos_arr1,
                               const std::vector<std::vector<uint8_t>>& split_bools,
                               const std::vector<size_t>& index_arr);

        void extract_bits_single(std::vector<std::vector<uint8_t>>& bool0_vec,
                                 std::vector<std::vector<uint8_t>>& bool1_vec,
                                 const std::vector<uint64_t>& source0,
                                 const std::vector<uint64_t>& source1,
                                 const std::vector<size_t>& positions,
                                 bool is_arr0);

        void extract_bits(std::vector<std::vector<uint8_t>>& bool0_vec,
                          std::vector<std::vector<uint8_t>>& bool1_vec,
                          const std::vector<uint64_t>& selfs0,
                          const std::vector<uint64_t>& peers0,
                          const std::vector<uint64_t>& selfs1,
                          const std::vector<uint64_t>& peers1,
                          const std::vector<size_t>& idx_pos_arr0,
                          const std::vector<size_t>& idx_pos_arr1);

        void process_both_non_empty(std::vector<std::vector<uint8_t>>& bool0_vec,
                                    std::vector<std::vector<uint8_t>>& bool1_vec,
                                    std::vector<uint64_t>& bool0,
                                    std::vector<uint64_t>& bool1,
                                    const std::vector<size_t>& idx_pos_arr0,
                                    const std::vector<size_t>& idx_pos_arr1);

        void process_single_non_empty(std::vector<std::vector<uint8_t>>& bool0_vec,
                                      std::vector<std::vector<uint8_t>>& bool1_vec,
                                      const std::vector<uint64_t>& bool0,
                                      const std::vector<uint64_t>& bool1,
                                      const std::vector<size_t>& idx_pos_arr0,
                                      const std::vector<size_t>& idx_pos_arr1);

        void prepare_bool_vectors(std::vector<uint64_t>& bool0, std::vector<uint64_t>& bool1,
                                  std::vector<size_t>& idx_pos_arr0, std::vector<size_t>& idx_pos_arr1,
                                  const std::vector<std::vector<uint8_t>>& split_bools,
                                  const std::vector<size_t>& index_arr);

        std::array<std::vector<std::vector<uint8_t>>, 2>
        split_bool_vec_from_bools(const std::vector<uint64_t>& selfs, const std::vector<uint64_t>& peers,
                                  const std::vector<size_t>& idx_pos_arr);

        void process_both_non_empty(std::vector<uint64_t>& bool0, std::vector<uint64_t>& bool1,
                                    std::vector<size_t>& idx_pos_arr0, std::vector<size_t>& idx_pos_arr1,
                                    std::vector<GradHess>& gh_ashr0, std::vector<GradHess>& gh_ashr1,
                                    size_t& comm_bytes);

        void process_single_array(bool is_arr0, std::vector<uint64_t>& bools,
                                  const std::vector<size_t>& idx_pos_arr,
                                  std::vector<GradHess>& gh_ashr0, std::vector<GradHess>& gh_ashr1,
                                  size_t& comm_bytes);

        void multiplex_negative(std::vector<uint64_t> &gshares, std::vector<uint64_t> &hshares,
                                const std::vector<uint8_t> &masks);

        void multiplex_positive(std::vector<uint64_t> &gshares, std::vector<uint64_t> &hshares,
                                const std::vector<uint8_t> &masks);

        void multiplex_zero(std::vector<uint64_t> &gshares, std::vector<uint64_t> &hshares,
                            const std::vector<uint8_t> &masks);

        std::shared_ptr<yacl::link::Context> link_ctx_;
        psi::PsiRoleType role_;

        std::shared_ptr<BasicOTProtocols> base_ot_;

        std::shared_ptr<VoleCPsi> vole_cpsi_ptr0_;
        std::shared_ptr<VoleCPsi> vole_cpsi_ptr1_;

        std::shared_ptr<HistoGram> histogram_ptr0_;
        std::shared_ptr<HistoGram> histogram_ptr1_;

        std::shared_ptr<Operators> operators_ptr_;

        std::vector<uint8_t> mask0_;
        std::vector<uint8_t> mask1_;

        GradHess grad_hess_ashr0_;
        GradHess grad_hess_ashr1_;

        std::vector<uint64_t> label_ashr0_;
        std::vector<uint64_t> label_ashr1_;

        std::vector<std::vector<uint64_t>> leaf_wt_ashr_vec_;
        std::vector<uint64_t> tree_eval_ashr0_;
        std::vector<uint64_t> tree_eval_ashr1_;

        std::vector<std::vector<uint8_t>> self_split_bools_;
        std::vector<std::vector<uint8_t>> split_bool_shrs_;

        size_t self_rows_ = 0;
        size_t peer_rows_ = 0;
        size_t self_cols_ = 0;
        size_t peer_cols_ = 0;

        size_t log2_scale_ = 25;

        std::shared_ptr<seal::SEALContext> bfv_context_;
        std::shared_ptr<seal::BatchEncoder> bfv_encoder_;
        seal::SecretKey secret_key_;

        std::shared_ptr<seal::Evaluator> bfv_evaluator_;
        std::shared_ptr<seal::Encryptor> bfv_encryptor_;
        std::shared_ptr<seal::Decryptor> bfv_decryptor_;

        const size_t bfv_coeff_count_ = 4096;
        const uint64_t bfv_plain_mod_ = 281474976694273; // approx 2^48

        bool is_inference_;
        bool enable_sigmoid_;
    };

    std::vector<std::shared_ptr<AnonXGB>> 
    initialize(std::vector<std::shared_ptr<yacl::link::Context>> &ctxs,
               const std::vector<std::string> &items, const std::vector<double> &labels,
               size_t cols, uint64_t log2_scale);
    
    std::vector<std::shared_ptr<AnonXGB>> initialize_inference(std::vector<std::shared_ptr<yacl::link::Context>> &ctxs,
                                                               const std::vector<std::string> &items, 
                                                               psi::PsiRoleType role,
                                                               size_t cols, 
                                                               uint64_t log2_scale);

    std::pair<std::vector<std::vector<size_t>>, std::vector<uint64_t>> 
    gen_tree_parallel(std::vector<std::shared_ptr<AnonXGB>> &ptr_vec,
                const std::vector<std::vector<uint8_t>> &split_bools, double lambda,
                int tree_level, bool is_last_tree);

    std::vector<double> infer_parallel(std::vector<std::shared_ptr<AnonXGB>> &ptr_vec,
                                       const size_t &num_tree, 
                                       const size_t &depth,
                                       std::vector<std::pair<std::vector<std::vector<std::tuple<int, double, double>>>,std::vector<uint64_t>>> &model,
                                       std::vector<std::vector<double>> &feats,
                                       bool enable_lookup_table_cal_weight=true);
}

#endif //SPU_ANON_XGB_H
